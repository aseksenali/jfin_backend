import argparse
import logging

from flask import Flask, request, jsonify

from jfin_gpt.constants import SOURCE_DIRECTORY, DEVICE_TYPE
from jfin_gpt.documents import documents_service
from jfin_gpt.exceptions import CollectionDoesNotExistException
from jfin_gpt.llm import llm_service
from jfin_gpt.milvus import milvus_service

logging.info(f"Running on: {DEVICE_TYPE}")
app = Flask(__name__)


@app.route("/api/heath", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


@app.route("/api/prompt", methods=["POST"])
async def prompt_route():
    body = request.json
    user_prompt = body.get('prompt')
    chat_history = body.get('chat_history')

    if user_prompt:
        logging.info(f'User Prompt: {user_prompt}')
        response = llm_service.ask_question(user_prompt, chat_history)
        logging.info(f'Generated response: {response}')

        prompt_response_dict = {
            "answer": response['answer'],
            "sources": []
        }

        return jsonify(prompt_response_dict), 200

    return jsonify({"error": "No user prompt provided."}), 400


@app.route("/api/documents", methods=["GET"])
def get_documents():
    return documents_service.get_files(SOURCE_DIRECTORY)


@app.route("/api/documents", methods=["POST"])
def add_document():
    if 'document' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['document']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            file_path, file_name, creation_date = documents_service.save_file(file)
            milvus_service.insert_file_or_directory(file_path)
            return jsonify(
                {'message': 'File saved successfully', 'creation_date': creation_date, 'file_name': file_name}), 200
        except FileExistsError:
            return jsonify({'error': 'File already exists.'}), 409
        except CollectionDoesNotExistException as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            return jsonify({'error': f'An error occurred while saving the file: {str(e)}'}), 500


@app.route('/api/documents/<filename>', methods=["DELETE"])
def delete_document(filename):
    if not documents_service.has_file(filename):
        return jsonify({"error": "File does not exist"}), 404

    try:
        documents_service.delete_file(filename)
        milvus_service.delete_document(filename)
        return jsonify({"message": "File deleted successfully", 'file_name': filename}), 200
    except CollectionDoesNotExistException as e:
        logging.error(str(e))
        return jsonify({'error': str(e)}), 404


@app.route("/api/sources", methods=["DELETE"])
def reset_milvus():
    documents_service.reset_sources()
    milvus_service.reset_documents()
    return jsonify({"message": "Milvus collection deleted"}), 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5110,
                        help="Port to run the UI on. Defaults to 5110.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the UI on. Defaults to 127.0.0.1. "
                             "Set to 0.0.0.0 to make the UI externally "
                             "accessible from other devices.")
    args = parser.parse_args()
    app.run(debug=False, host=args.host, port=args.port)
