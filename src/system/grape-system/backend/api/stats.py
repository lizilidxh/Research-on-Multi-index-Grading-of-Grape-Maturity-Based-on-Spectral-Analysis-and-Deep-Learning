# =============================================================================
#  系统统计接口
# =============================================================================
import traceback
import pandas as pd
from flask import jsonify
from backend.config import DATA_DIR, NAMES

def register_stats_routes(app, mgr):
    @app.route('/api/stats')
    def stats():
        try:
            df = pd.read_csv(f'{DATA_DIR}/指标数据_含标签.csv')
            dist = df['maturity_label'].value_counts().sort_index().to_dict()
            return jsonify({
                'total_samples': len(df),
                'distribution': {NAMES[k]: dist.get(k,0) for k in range(4)},
                'model_metrics': {
                    'spectrum_accuracy': 0.8773,
                    'spectrum_f1': 0.8775,
                    'spectrum_cv': '0.8736±0.0112',
                    'image_accuracy': round(mgr.img_acc,4),
                    'image_cv': f'{mgr.img_cv_mean:.4f}±{mgr.img_cv_std:.4f}',
                }
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500