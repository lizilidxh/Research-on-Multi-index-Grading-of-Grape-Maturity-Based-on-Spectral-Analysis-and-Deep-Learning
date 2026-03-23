# =============================================================================
#  图像预测接口
# =============================================================================
import time
import traceback
from flask import request, jsonify
from backend.config import ALLOWED_IMAGE_EXTENSIONS

def register_image_predict_routes(app, mgr):
    @app.route('/api/predict/image', methods=['POST'])
    def predict_image():
        """图像预测：上传一张葡萄图片"""
        if 'image' not in request.files:
            return jsonify({'error': '请上传图片文件（字段名: image）'}), 400

        file = request.files['image']
        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            return jsonify({'error': f'仅支持 {",".join(ALLOWED_IMAGE_EXTENSIONS)} 格式'}), 400

        try:
            t0 = time.time()
            res = mgr.predict_image(file.read())
            res['elapsed_ms'] = round((time.time()-t0)*1000, 1)
            res['success'] = True
            return jsonify(res)
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500