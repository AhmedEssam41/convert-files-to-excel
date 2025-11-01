import os
import re
import base64
import requests
import pandas as pd
import pdfplumber
import docx
from flask import Flask, request, render_template, send_file, jsonify

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "outputs"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# ------------------------
# قراءة PDF
# ------------------------
def read_pdf(file_path):
    dfs = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                df = pd.DataFrame(table[1:], columns=table[0])
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None

# ------------------------
# قراءة Word
# ------------------------
def read_docx(file_path):
    doc = docx.Document(file_path)
    dfs = []
    for table in doc.tables:
        data, keys = [], None
        for i, row in enumerate(table.rows):
            text = [cell.text.strip() for cell in row.cells]
            if i == 0:
                keys = text
            else:
                data.append(text)
        if keys:
            dfs.append(pd.DataFrame(data, columns=keys))
    return pd.concat(dfs, ignore_index=True) if dfs else None

# ------------------------
# قراءة صورة باستخدام Qwen
# ------------------------

def read_image_with_qwen(file_path, required_columns=None):
    with open(file_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    if len(required_columns) == 1:
        user_prompt = f"what is the {required_columns[0]} in this image?"
    else:
        columns = ",".join(required_columns)
        user_prompt = f"""
          : حلل هذه الصورة و تحقق من جميع الحقول و المعلومات التي تراها ثم أخبرني بقيم الحقول التاليه
          {{{columns}}}
          إذا كان اسم أي حقل غير منطقي أو لا توجد له بيانات في الصورة أرجع قيمته فارغة"".
        """

  # بيانات الاتصال بـ Hugging Face API
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # 🔹 حط هنا الـ Access Token بتاعك من https://huggingface.co/settings/tokens
    MODEL_URL = "https://api-inference.huggingface.co/models/ahmed-20033/my-ai-model"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    payload = {
        "inputs": {
            "text": user_prompt,
            "image": f"data:image/jpeg;base64,{image_b64}"
        }
    }

    try:
        response = requests.post(MODEL_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        text = result[0].get("generated_text", result[0] if isinstance(result, list) else result)
        if isinstance(text, dict):
            text = text.get("generated_text", "")
        text = str(text).strip()
        print("📜 Raw Qwen response:", text)  # للتصحيح
       # 1️⃣ هات كل النصوص اللي بين الأقواس { }
        inside_braces = re.findall(r'\{([^}]*)\}', text, flags=re.MULTILINE)
        if inside_braces:
            # لو لقى أقواس، خُد اللي جواها بس
            result = inside_braces
        else:
            # 2️⃣ لو مفيش، هات السطور اللي تبدأ بـ -
            result = re.findall(r'^-.*', text, flags=re.MULTILINE)

        block = "\n".join(result)   # دمج كل العناصر في نص واحد
        matches = re.findall(r'^[^:]*:\s*(.+)$', block, flags=re.MULTILINE)
        # تنظيف:
        values = [
            x.strip(' "\',')    # يشيل أى مسافات أو " أو ' أو ,
            for x in matches
            if x.strip(' "\',') != ''
        ]

        print("📜 Processed model_text:", values)  # للتصحيح

        if values == []:
            raise ValueError("لم يتم استخراج أي نصوص من الصورة.")

        if required_columns:
            if any("لا يوجد" in v or "خطأ" in v or "تبحث" in v or "لا تحتوي" in v for v in values):
                raise ValueError(
                    f"النص المستخرج يحتوي على رسالة خطأ"
                )
            # نجمع العناصر تاني في نص واحد
            # joined = ",".join(required_columns)
            # نحولها لـ list حقيقية
            required_columns2 = required_columns
            # 1️⃣ شيل كل الأسماء المطلوبة (في النص كله)
            pattern = r"\b(?:%s)\b" % "|".join(map(re.escape, required_columns2))
            # امشي على كل عنصر (سطر) وعدلّه
            cleaned_lines = []
            for line in values:
                line_no_names = re.sub(pattern, "", line)
                # نظّف الفواصل والمسافات
                cleaned_line = ", ".join(
                    item.strip() for item in line_no_names.split(",") if item.strip()
                )
                cleaned_lines.append(cleaned_line)
            cleaned_lines = [line for line in cleaned_lines if line.strip()]

            if len(cleaned_lines) != len(required_columns2):
                raise ValueError(
                    f"عدد القيم المستخرجة ({len(cleaned_lines)}) لا يتطابق مع عدد الأعمدة المطلوبة ({len(required_columns2)}). النص المستخرج: {cleaned_lines}"
                )
            df = pd.DataFrame([cleaned_lines], columns=required_columns) 
            return df
        else:
            return pd.DataFrame([{"Text": text}])

    except Exception as e:
        raise ValueError(f"❌ خطأ في الاتصال بـ Qwen أو استخراج البيانات: {e}")

# ------------------------
# قراءة أي ملف
# ------------------------
def read_file(input_file, required_columns=None):
    if input_file.lower().endswith(".csv"):
        return pd.read_csv(input_file)
    elif input_file.lower().endswith(".xlsx"):
        return pd.read_excel(input_file)
    elif input_file.lower().endswith(".txt"):
        try:
            return pd.read_csv(input_file, delimiter=",")
        except:
            return pd.read_csv(input_file, delimiter="\t")
    elif input_file.lower().endswith(".pdf"):
        return read_pdf(input_file)
    elif input_file.lower().endswith(".docx"):
        return read_docx(input_file)
    elif input_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        return read_image_with_qwen(input_file, required_columns)
    else:
        raise ValueError("نوع الملف غير مدعوم")

# ------------------------
# صفحة HTML
# ------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ------------------------
# API لعرض النتائج في الموقع
# ------------------------
@app.route("/process", methods=["POST"])
def process():
    file = request.files["file"]
    required_columns = [col.strip() for col in request.form["columns"].split(",") if col.strip()]

    if not file:
        return jsonify({"error": "❌ من فضلك ارفع ملف."}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        df = read_file(file_path, required_columns)
        if required_columns:
            available_cols = []
            for col in required_columns:
                for df_col in df.columns:
                    if col.strip().lower() == df_col.strip().lower():
                        available_cols.append(df_col)

            # ✅ لو مفيش أي تطابق بين الأعمدة المطلوبة واللي اتقرت
            if not available_cols:
                return jsonify({
                    "error": "❌ لم يتم العثور على أي من الأعمدة المطلوبة داخل الملف/الصورة."
                }), 400
        
        # فحص إضافي: التأكد إن البيانات تحتوي على نصوص مستخرجة
        if df.empty or df.dropna().empty:
            return jsonify({
                "error": "❌ لم يتم استخراج أي بيانات من الملف أو الصورة."
            }), 400

        df = df[available_cols]

        # تنظيف أسماء الأعمدة
        df.columns = [re.sub(r'["\[\]]', '', col) for col in df.columns]

        # نخزن نسخة Excel عشان التحميل
        output_file = os.path.join(app.config["OUTPUT_FOLDER"], file.filename.rsplit(".", 1)[0] + "_filtered.xlsx")
        df.to_excel(output_file, index=False)

        return jsonify({
            "columns": list(df.columns),
            "rows": df.to_dict(orient="records"),
            "download_url": f"/download/{os.path.basename(output_file)}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------
# API لتحميل Excel
# ------------------------
@app.route("/download/<filename>")
def download(filename):
    file_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "❌ الملف غير موجود", 404

@app.route("/health")
def health_check():
    return "OK", 200

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)  
