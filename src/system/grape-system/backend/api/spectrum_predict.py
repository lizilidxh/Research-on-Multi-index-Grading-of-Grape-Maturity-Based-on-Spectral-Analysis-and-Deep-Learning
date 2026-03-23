# =============================================================================
#  高光谱预测接口
# =============================================================================
import time
import traceback
import numpy as np
import pandas as pd
from io import StringIO
from flask import request, jsonify
from collections import Counter
from backend.config import NAMES, PALETTE
from backend.utils import get_wavelength_columns


def register_spectrum_predict_routes(app, mgr):
    @app.route('/api/predict/file', methods=['POST'])
    def predict_file():
        """高光谱CSV文件预测"""
        if not mgr.spec_loaded:
            return jsonify({'error': '高光谱模型未就绪'}), 503
        if 'file' not in request.files:
            return jsonify({'error': '请上传 CSV 文件'}), 400

        file = request.files['file']
        try:
            df = pd.read_csv(StringIO(file.read().decode('utf-8-sig')))
            wave_cols = get_wavelength_columns(df)
            X_raw = df[wave_cols].values.astype(float)
            X_raw = np.nan_to_num(X_raw, nan=np.nanmean(X_raw))

            t0 = time.time()
            labels, probs = mgr.predict_spectrum(X_raw)
            elapsed = time.time() - t0

            dist = dict(Counter(labels))
            results = []
            ids = df['序号'].tolist() if '序号' in df.columns else list(range(1, len(labels) + 1))

            for i, (lb, pr) in enumerate(zip(labels, probs)):
                results.append({
                    'sample_id': ids[i], 'label': lb,
                    'name': NAMES[lb], 'color': PALETTE[lb],
                    'confidence': round(float(pr.max()), 4),
                    'probabilities': {NAMES[j]: round(float(pr[j]), 4) for j in range(4)},
                })

            return jsonify({
                'success': True, 'count': len(results),
                'elapsed_ms': round(elapsed * 1000, 1),
                'distribution': {NAMES[k]: dist.get(k, 0) for k in range(4)},
                'results': results,
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500