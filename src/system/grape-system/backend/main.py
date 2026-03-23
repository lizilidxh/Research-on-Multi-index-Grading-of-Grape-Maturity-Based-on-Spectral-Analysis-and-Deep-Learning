# =============================================================================
#  葡萄成熟度识别系统 v2 —— 主程序入口
#  运行：python main.py → http://localhost:5000
# =============================================================================
import warnings
from flask import Flask
from config import BACKEND_DIR
from utils import add_cors_headers, options_handler
from models import SystemManager
from api import (
    register_health_routes,
    register_image_predict_routes,
    register_spectrum_predict_routes,
    register_stats_routes
)

# 忽略警告
warnings.filterwarnings('ignore')

# 初始化Flask应用
app = Flask(__name__, root_path=BACKEND_DIR)

# 注册CORS处理
app.after_request(add_cors_headers)
app.add_url_rule('/', defaults={'path': ''}, view_func=options_handler, methods=['OPTIONS'])
app.add_url_rule('/<path:path>', view_func=options_handler, methods=['OPTIONS'])

# 初始化模型管理器
mgr = SystemManager()

# 注册所有API路由
register_health_routes(app, mgr)
register_image_predict_routes(app, mgr)
register_spectrum_predict_routes(app, mgr)
register_stats_routes(app, mgr)

if __name__ == '__main__':
    print('\n' + '=' * 56)
    print('  葡萄成熟度识别系统 v2 · 图像+高光谱双模式')
    print('=' * 56)

    # 加载所有模型
    mgr.load_all()

    print(f'\n  服务启动 → http://localhost:5000')
    app.run(debug=False, host='0.0.0.0', port=5000)