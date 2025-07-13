import os
import cv2
from yolov8_detector import load_model, detect_objects
from ocr import run_ocr
from levenshtein import calculate_levenshtein_metrics
from llm import query_llm
from spreadsheet import save_results_to_excel

def process_detection(model, image, image_path, class_name, confidence):
    print(f"\n🔍 Running OCR and analysis for: {os.path.basename(image_path)}")

    ocr_results = run_ocr(image_path)
    avg_sim, avg_dist, lev_adj = calculate_levenshtein_metrics(class_name, ocr_results)
    yolo_plus_lev = min(0.999, confidence + avg_sim)

    llm_reply, llm_adj = query_llm(class_name, ocr_results)
    final_conf_llm = min(0.999, max(0.001, confidence + llm_adj))

    print(f"📏 Avg Levenshtein distance: {avg_dist:.3f}")
    print(f"🧠 Avg Levenshtein similarity: {avg_sim:.3f}")
    print(f"💬 LLM response:\n{llm_reply}")
    print(f"✅ Final accuracy with LLM: {final_conf_llm:.3f}")

    return {
        "YOLO Class": class_name,
        "YOLO Confidence": round(confidence, 3),
        "OCR Words": ocr_results,
        "Avg Levenshtein Distance": round(avg_dist, 3),
        "Avg Levenshtein Similarity": round(avg_sim, 3),
        "YOLO + Levenshtein": round(yolo_plus_lev, 3),
        "LLM Response": llm_reply,
        "Final Accuracy with LLM": round(final_conf_llm, 3),
        "Below 70%": "Yes" if confidence < 0.7 else "No"
    }

def run_pipeline(model_path, image_folder, output_excel_path):
    model, device = load_model(model_path)
    os.makedirs(image_folder, exist_ok=True)

    image_files = [f for f in sorted(os.listdir(image_folder)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    total_images = len(image_files)
    results = []

    for idx, filename in enumerate(image_files, start=1):
        print(f"\n🖼️ Processing image: {filename} ({idx}/{total_images})")
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        result = detect_objects(model, image, device)
        annotated = result.plot()
        annotated_path = os.path.join(image_folder, f"annotated_{filename}")
        cv2.imwrite(annotated_path, annotated)

        for j, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            class_name = model.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = image[y1:y2, x1:x2]
            cropped_path = os.path.join(image_folder, f"cropped_{filename[:-4]}_{j}.png")
            cv2.imwrite(cropped_path, cropped)

            result_data = process_detection(model, image, cropped_path, class_name, confidence)
            result_data["Original Image"] = filename
            result_data["Annotated Image"] = annotated_path

            results.append(result_data)

        save_results_to_excel(results, output_excel_path)
        progress = (idx / total_images) * 100
        print(f"📊 Progress: {progress:.2f}%")

if __name__ == "__main__":
    model_path = "./Weights 3/best.pt"
    image_folder = "./Images/01 - Tests/Test 03"
    output_excel_path = "./Images/01 - Tests/Test 03/Results/results_analysis.xlsx"
    run_pipeline(model_path, image_folder, output_excel_path)
