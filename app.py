from flask import Flask, request, render_template, send_from_directory
import ollama
import os
import subprocess
import glob
import json
import tempfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['FRAMES_FOLDER'] = 'extracted_frames/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FRAMES_FOLDER'], exist_ok=True)

def cleanup_frames_folder():
    """Clean up all files in the frames folder"""
    frames_folder = app.config['FRAMES_FOLDER']
    if os.path.exists(frames_folder):
        for filename in os.listdir(frames_folder):
            file_path = os.path.join(frames_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        print("Frames folder cleaned up")


def get_prompt(category):
    if category == 'in_cabin':
        return (
            "You are an AI vision safety analyzer. "
            "Look at the image and provide:\n\n"
            "1. A short natural language summary of what you see (2–3 sentences).\n"
            "2. Then output STRICT JSON only, with keys: eating, sleeping, phone_use, fatigue, extra_passengers. "
            "Each key must be an object with {\"value\": true/false, \"confidence\": float}.\n\n"
            "Example:\n"
            "Summary: The driver appears to be holding a phone while driving.\n"
            "{\n"
            "  \"eating\": {\"value\": false, \"confidence\": 0.1},\n"
            "  \"sleeping\": {\"value\": false, \"confidence\": 0.05},\n"
            "  \"phone_use\": {\"value\": true, \"confidence\": 0.92},\n"
            "  \"fatigue\": {\"value\": false, \"confidence\": 0.15},\n"
            "  \"extra_passengers\": {\"value\": false, \"confidence\": 0.0}\n"
            "}\n"
            "Important: Always include BOTH the summary and the JSON, in that order."
        )
    else:
        return (
            "You are an AI vision safety analyzer. "
            "Look at the road-facing image and provide:\n\n"
            "1. A short natural language summary of what you see (2–3 sentences).\n"
            "2. Then output STRICT JSON only, with keys: accident_risk, weather, hazards. "
            "Each key must be an object with {\"value\": true/false or string, \"confidence\": float}.\n\n"
            "Example:\n"
            "Summary: The road is wet and visibility is reduced due to rain.\n"
            "{\n"
            "  \"accident_risk\": {\"value\": true, \"confidence\": 0.75},\n"
            "  \"weather\": {\"value\": \"rain\", \"confidence\": 0.88},\n"
            "  \"hazards\": {\"value\": true, \"confidence\": 0.65}\n"
            "}\n"
            "Important: Always include BOTH the summary and the JSON, in that order."
        )


def analyze_image(image_path, category):
    prompt = get_prompt(category)
    response = ollama.chat(
        model="llama3.2-vision:11b",
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_path]
        }]
    )

    raw_content = response["message"]["content"]

    # Split summary vs JSON
    summary = ""
    parsed = {}
    if "{" in raw_content and "}" in raw_content:
        try:
            summary = raw_content.split("{")[0].replace("Summary:", "").strip()
            json_part = raw_content[raw_content.index("{"): raw_content.rindex("}")+1]
            parsed = json.loads(json_part)
        except Exception:
            parsed = {"error": "Could not parse model output", "raw": raw_content}
    else:
        summary = raw_content

    return {"summary": summary, "flags": parsed}


def extract_frames(video_path, fps=1):
    """Extract frames from video using ffmpeg to permanent folder"""
    # Clean up previous frames
    cleanup_frames_folder()
    
    frames_folder = app.config['FRAMES_FOLDER']
    frame_pattern = os.path.join(frames_folder, "frame_%04d.png")
    
    print(f"Extracting frames to: {frames_folder}")
    cmd = [
        "ffmpeg", "-i", video_path, "-vf", f"fps={fps}",
        frame_pattern, "-hide_banner", "-loglevel", "error"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        extracted_frames = sorted(glob.glob(os.path.join(frames_folder, "*.png")))
        print(f"Extracted {len(extracted_frames)} frames")
        return extracted_frames
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        raise Exception(f"Video processing failed: {e}")
    except Exception as e:
        print(f"Frame extraction error: {e}")
        raise


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("POST request received")
        print("Files:", request.files)
        print("Form data:", request.form)
        
        file = request.files.get('file')
        category = request.form.get('category')
        fps = int(request.form.get('fps', 1))

        print(f"File object: {file}")
        print(f"File filename: {file.filename if file else 'No file'}")
        print(f"Category: {category}")
        print(f"FPS: {fps}")

        if not file or file.filename == '':
            print("No file uploaded - returning error")
            return render_template('index.html', error="No file uploaded")

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        print(f"Saving file to: {filepath}")
        file.save(filepath)

        mimetype = file.mimetype.lower()
        print(f"File mimetype: {mimetype}")
        results = []
        summary_flags = {}

        try:
            print("Starting analysis...")
            # Clean frames folder at the start of each analysis
            cleanup_frames_folder()
            
            if "video" in mimetype:
                print("Processing as video")
                frames = extract_frames(filepath, fps=fps)
                for i, frame in enumerate(frames):
                    print(f"Analyzing frame {i+1}/{len(frames)}: {frame}")
                    results.append(analyze_image(frame, category))
            else:
                print("Processing as image")
                results.append(analyze_image(filepath, category))
            print("Analysis completed successfully")

            # Aggregate flags (OR + max confidence across frames)
            for res in results:
                flags = res.get("flags", {})
                for k, v in flags.items():
                    if isinstance(v, dict) and "value" in v:
                        if k not in summary_flags:
                            summary_flags[k] = v
                        else:
                            summary_flags[k]["value"] = summary_flags[k]["value"] or v["value"]
                            summary_flags[k]["confidence"] = max(
                                summary_flags[k]["confidence"], v["confidence"]
                            )

        except Exception as e:
            print(f"Analysis failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template('index.html', error=f"Analysis failed: {str(e)}")

        return render_template(
            'result.html',
            filename=file.filename,
            category=category,
            results=results,
            summary_flags=summary_flags,
            flags=summary_flags,
            summary=results[0]["summary"] if results else "No summary",
            raw_json=results[0]["flags"] if results else {}
        )

    return render_template('index.html')


# serve uploads folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# serve extracted frames folder for debugging
@app.route('/frames/<filename>')
def extracted_frame(filename):
    return send_from_directory(app.config['FRAMES_FOLDER'], filename)


if __name__ == '__main__':
    print("Starting Image Safety App...")
    print("Cleaning up frames folder on startup...")
    cleanup_frames_folder()
    print("App ready!")
    app.run(debug=True)
