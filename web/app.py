import os
import io
import json
import uuid
import base64
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, jsonify, make_response
import torch
import numpy as np
from PIL import Image
import cv2

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.diagnosis.concept_bottleneck import FuzzyConceptBottleneckModel
from src.counterfactual.concept_intervention import ConceptInterventionEngine
from src.counterfactual.counterfactual_generator import CounterfactualGenerator
from src.counterfactual.explanation_report import ExplanationReport
from src.preprocessing.artefact_removal import remove_hair_dullrazor
from src.preprocessing.illumination import normalize_illumination_gray_world
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.segmentation.unet_baseline import UNetBaseline

# --- AWS Integration ---
try:
    from web.aws_manager import AWSManager, require_cognito_auth
    from web.db_models import get_db_engine, init_db, get_session, DiagnosisRecord
    AWS_ENABLED = True
except ImportError as e:
    print(f"Warning: AWS modules not found or failed to load. ({e})")
    AWS_ENABLED = False
    def require_cognito_auth(getter):
        def dec(f): return f
        return dec

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'}
ALLOWED_MIMETYPES = {'image/jpeg', 'image/png', 'image/bmp', 'image/tiff'}
MIN_IMAGE_DIMENSION = 32


def validate_upload(file):
    """Validate uploaded image file for extension, MIME type, dimensions, and corruption.

    Returns:
        Tuple of (is_valid, error_message, pil_image_or_none)
    """
    if file is None or file.filename == '':
        return False, "No file selected.", None

    # Check extension
    filename = file.filename.lower()
    ext = filename.rsplit('.', 1)[-1] if '.' in filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '.{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}", None

    # Check MIME type
    if file.content_type and file.content_type not in ALLOWED_MIMETYPES:
        return False, f"Unsupported MIME type '{file.content_type}'.", None

    # Read and validate image
    try:
        img_bytes = file.read()
        if len(img_bytes) == 0:
            return False, "Uploaded file is empty.", None

        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = pil_img.size
        if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
            return False, f"Image too small ({w}×{h}). Minimum dimension: {MIN_IMAGE_DIMENSION}px.", None

        return True, "", pil_img
    except Exception as e:
        return False, f"Cannot read image file: {str(e)}. File may be corrupt.", None


@app.errorhandler(413)
def handle_file_too_large(e):
    """Handle file size exceeding MAX_CONTENT_LENGTH."""
    return jsonify({
        "error": "File too large",
        "message": "Maximum upload size is 16 MB.",
        "error_code": "FILE_TOO_LARGE"
    }), 413


@app.errorhandler(415)
def handle_unsupported_media(e):
    """Handle unsupported media type."""
    return jsonify({
        "error": "Unsupported media type",
        "message": "Only image files (JPG, PNG, BMP, TIFF) are accepted.",
        "error_code": "UNSUPPORTED_MEDIA"
    }), 415


@app.errorhandler(500)
def handle_internal_error(e):
    """Handle internal server errors."""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again.",
        "error_code": "INTERNAL_ERROR"
    }), 500

# --- Lazy Model & AWS Loading ---
MODEL = None
ENGINE = None
CF_GEN = None
UNET_MODEL = None
AWS_MGR = None
DB_SESSION = None

def get_services():
    global MODEL, ENGINE, CF_GEN, UNET_MODEL, AWS_MGR, DB_SESSION
    if MODEL is None:
        MODEL = FuzzyConceptBottleneckModel(num_classes=7)
        try:
            MODEL.load_state_dict(torch.load(PROJECT_ROOT / "models" / "concept_bottleneck.pth", map_location="cpu"))
        except Exception:
            print("Warning: Could not load pretrained weights. Using initialized weights.")
        MODEL.eval()
        ENGINE = ConceptInterventionEngine(MODEL)
        CF_GEN = CounterfactualGenerator(MODEL)

    if UNET_MODEL is None:
        unet_path = PROJECT_ROOT / "models" / "segmentation_unet_baseline.pt"
        if unet_path.exists():
            try:
                UNET_MODEL = UNetBaseline(in_channels=3, out_channels=1)
                ckpt = torch.load(unet_path, map_location="cpu")
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                UNET_MODEL.load_state_dict(state_dict)
                UNET_MODEL.eval()
                print("Loaded U-Net segmentation checkpoint successfully.")
            except Exception as e:
                print(f"Notice: U-Net weights failed to load ({e}). Using Otsu fallback.")
                UNET_MODEL = None

    if AWS_ENABLED and AWS_MGR is None:
        try:
            AWS_MGR = AWSManager()
            db_creds = AWS_MGR.get_db_secret()
            if db_creds:
                db_engine = get_db_engine(db_creds)
                init_db(db_engine)
                DB_SESSION = get_session(db_engine)
        except Exception as e:
            print(f"AWS Setup Notice: {e}. Running in local mode.")

    return MODEL, ENGINE, CF_GEN, AWS_MGR, DB_SESSION


def segment_lesion(image_np):
    """Segment lesion using trained U-Net if available, otherwise Otsu thresholding."""
    global UNET_MODEL
    if UNET_MODEL is not None:
        try:
            img_t = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            with torch.no_grad():
                pred = UNET_MODEL(img_t)
                prob = torch.sigmoid(pred).squeeze().numpy()
                mask = (prob > 0.5).astype(np.uint8)
                if np.sum(mask) > 10:
                    return mask
        except Exception as e:
            print(f"U-Net inference exception: {e}. Falling back to Otsu.")

    # Otsu fallback with morphological post-processing
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return (mask > 127).astype(np.uint8)


def img_to_base64(img_np):
    img_pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
@require_cognito_auth(get_services)
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded", "error_code": "NO_IMAGE"}), 400

    file = request.files["image"]

    # Validate upload
    is_valid, error_msg, img_pil = validate_upload(file)
    if not is_valid:
        return jsonify({"error": error_msg, "error_code": "INVALID_UPLOAD"}), 400

    img_pil = img_pil.resize((256, 256))
    img_np = np.array(img_pil)
    img_bytes = None
    # Re-read for S3 upload if needed
    file.seek(0)
    img_bytes = file.read()

    # Optional preprocessing
    apply_preprocess = request.form.get("preprocess", "true").lower() == "true"
    if apply_preprocess:
        try:
            cleaned_np = remove_hair_dullrazor(img_np)
            cleaned_np = normalize_illumination_gray_world(cleaned_np)
        except Exception:
            cleaned_np = img_np
    else:
        cleaned_np = img_np

    mask_np = segment_lesion(cleaned_np)
    model, _, _, aws_mgr, db_session = get_services()

    try:
        pred_res = model.predict_image(cleaned_np, mask_np)
    except Exception as e:
        if aws_mgr: aws_mgr.logger.error(f"Prediction Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

    overlay = img_np.copy()
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    concept_list = pred_res["concept_vector"].tolist()

    # AWS Integrations: Upload to S3 and Log to RDS CloudWatch
    s3_img_url, s3_mask_url = None, None
    if aws_mgr:
        aws_mgr.log_inference(f"Successful inference for predicted class {pred_res['predicted_class_name']}")

        # Upload images to S3
        unique_id = str(uuid.uuid4())
        s3_img_url = aws_mgr.upload_image_to_s3(img_bytes, f"uploads/img_{unique_id}.jpg")

        # Prepare Mask for S3
        mask_pil = Image.fromarray(mask_np * 255)
        m_buf = io.BytesIO()
        mask_pil.save(m_buf, format="PNG")
        s3_mask_url = aws_mgr.upload_image_to_s3(m_buf.getvalue(), f"masks/mask_{unique_id}.png")

        # Save to RDS
        if db_session:
            record = DiagnosisRecord(
                original_image_s3_url=s3_img_url,
                mask_image_s3_url=s3_mask_url,
                asymmetry_score=concept_list[0],
                border_irregularity=concept_list[1],
                color_variation=concept_list[2],
                diameter_score=concept_list[3],
                predicted_class=pred_res["predicted_class_name"],
                confidence=pred_res["confidence"],
                class_probabilities=pred_res["class_probabilities"]
            )
            db_session.add(record)
            db_session.commit()

    return jsonify({
        "original_img_b64": img_to_base64(img_np),
        "preprocessed_img_b64": img_to_base64(cleaned_np),
        "overlay_img_b64": img_to_base64(overlay),
        "mask_img_b64": img_to_base64((mask_np * 255).astype(np.uint8)),
        "features": pred_res["extracted_features"],
        "fuzzy_concepts": pred_res["fuzzy_concepts"],
        "concept_vector": concept_list,
        "predicted_class": pred_res["predicted_class_name"],
        "confidence": pred_res["confidence"],
        "probabilities": pred_res["class_probabilities"]
    })


@app.route("/api/intervene", methods=["POST"])
@require_cognito_auth(get_services)
def intervene():
    data = request.json
    orig_c = torch.tensor([data["concepts"]], dtype=torch.float32)
    interventions = {int(k): float(v) for k, v in data["interventions"].items()}

    _, engine, _, aws_mgr, _ = get_services()
    int_res = engine.intervene_and_predict(orig_c, interventions=interventions)

    if aws_mgr:
        aws_mgr.log_inference(f"Intervention requested. New diagnosis: {int_res['intervened_diagnosis']}")

    return jsonify({
        "intervened_diagnosis": int_res["intervened_diagnosis"],
        "intervened_probabilities": int_res["intervened_probabilities"]
    })


@app.route("/api/counterfactual", methods=["POST"])
@require_cognito_auth(get_services)
def counterfactual():
    data = request.json or {}
    concepts = data.get("concepts")
    if not concepts or not isinstance(concepts, (list, tuple)) or len(concepts) != 4:
        return jsonify({
            "error": "Valid concept vector with 4 features is required. Please upload and analyze an image first.",
            "error_code": "INVALID_CONCEPTS"
        }), 400

    target_raw = data.get("target_class")
    if target_raw is None:
        return jsonify({
            "error": "Target class index is required.",
            "error_code": "MISSING_TARGET_CLASS"
        }), 400

    try:
        target_idx = int(target_raw)
        if not (0 <= target_idx <= 6):
            raise ValueError(f"Target class index {target_idx} out of range [0, 6]")
        orig_c = torch.tensor([concepts], dtype=torch.float32)
    except (ValueError, TypeError) as e:
        return jsonify({
            "error": f"Invalid input parameters: {e}",
            "error_code": "BAD_PARAMETERS"
        }), 400

    try:
        _, _, cf_gen, aws_mgr, _ = get_services()
        cf_res = cf_gen.generate_counterfactual_concept(orig_c, target_class_idx=target_idx)
    except Exception as e:
        return jsonify({
            "error": f"Counterfactual optimization failed: {e}",
            "error_code": "OPTIMIZATION_FAILED"
        }), 500

    # Render visual counterfactual lesion image if original image is provided
    cf_img_b64 = None
    if "original_image_b64" in data and data["original_image_b64"]:
        try:
            raw_b64 = data["original_image_b64"]
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(raw_b64)
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((256, 256))
            img_np = np.array(img_pil)
            mask_np = segment_lesion(img_np)
            cf_img_np, cf_mask_np = cf_gen.render_counterfactual_image(
                img_np, mask_np, cf_res["counterfactual_concepts"]
            )
            cf_img_b64 = img_to_base64(cf_img_np)
        except Exception as e:
            print(f"Counterfactual image rendering notice: {e}")

    if aws_mgr:
        try:
            aws_mgr.log_inference(f"Counterfactual generated for target {cf_res['target_class_name']}")
        except Exception as e:
            print(f"Notice: Telemetry log to CloudWatch skipped: {e}")

    return jsonify({
        "success": cf_res["success"],
        "target_class_name": cf_res["target_class_name"],
        "explanation_text": cf_res["explanation_text"],
        "counterfactual_concepts": cf_res["counterfactual_concepts"],
        "counterfactual_probabilities": cf_res["counterfactual_probabilities"],
        "counterfactual_image_b64": cf_img_b64
    })


@app.route("/api/report", methods=["POST"])
@require_cognito_auth(get_services)
def generate_report():
    """Generates structured clinical diagnostic report metadata."""
    data = request.json or {}
    report_id = f"RPT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

    report = {
        "report_id": report_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "system": "Fuzzy Concept-Bottleneck Skin Cancer Diagnosis v2.0",
        "predicted_class": data.get("predicted_class", "N/A"),
        "confidence": data.get("confidence", 0.0),
        "fuzzy_concepts": data.get("fuzzy_concepts", {}),
        "features": data.get("features", {}),
        "probabilities": data.get("probabilities", {}),
        "counterfactual_summary": data.get("counterfactual_summary", "No counterfactual requested."),
        "disclaimer": "RESEARCH PROTOTYPE ONLY: This diagnostic report is generated by an experimental AI model for investigational and research purposes only. It is NOT FDA-approved or cleared for primary clinical diagnosis. All findings must be validated by a certified dermatologist or pathologist."
    }
    return jsonify(report)


@app.route("/api/report/download", methods=["POST"])
@require_cognito_auth(get_services)
def download_report():
    """Generate and return a diagnostic report as a downloadable JSON file."""
    data = request.json or {}

    report_obj = ExplanationReport()
    # Populate from submitted data
    prediction_result = {
        "extracted_features": data.get("features", {}),
        "fuzzy_concepts": data.get("fuzzy_concepts", {}),
        "predicted_class_name": data.get("predicted_class", "N/A"),
        "confidence": data.get("confidence", 0.0),
        "class_probabilities": data.get("probabilities", {}),
    }
    report_obj.populate(
        prediction_result=prediction_result,
        counterfactual_result=data.get("counterfactual_summary"),
    )

    structured = report_obj.generate_structured_report()
    response = make_response(json.dumps(structured, indent=2, default=str))
    response.headers["Content-Type"] = "application/json"
    response.headers["Content-Disposition"] = (
        f"attachment; filename=diagnostic_report_{structured['report_id']}.json"
    )
    return response


@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({"error": str(e), "error_code": "BAD_REQUEST"}), 400


@app.errorhandler(500)
def handle_server_error(e):
    return jsonify({"error": "Internal server error occurred.", "error_code": "INTERNAL_ERROR"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
