# =============================================================================
#  健康检查接口
# =============================================================================
from flask import jsonify

def register_health_routes(app, mgr):
    @app.route('/api/health')
    def health():
        return jsonify({
            'status': 'ok',
            'spec_model': mgr.spec_loaded,
            'img_model': mgr.img_loaded,
            'wavelengths': len(mgr.wavelengths) if mgr.wavelengths else 0,
            'img_accuracy': round(mgr.img_acc, 4),
            'img_cv': f'{mgr.img_cv_mean:.4f}±{mgr.img_cv_std:.4f}',
            'model_info': {
                'image': 'Stacking(RF+GBM+SVM) → 38维颜色纹理特征',
                'spectrum': 'Stacking(SVM+RF+GBM+BP) → SG+SNV+PCA预处理',
            }
        })