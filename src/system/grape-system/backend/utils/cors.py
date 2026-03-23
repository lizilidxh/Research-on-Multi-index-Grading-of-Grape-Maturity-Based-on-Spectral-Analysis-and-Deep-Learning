# =============================================================================
#  跨域请求处理工具
# =============================================================================
from flask import make_response

def add_cors_headers(response):
    """为响应添加CORS头"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

def options_handler(path=''):
    """处理OPTIONS预检请求"""
    r = make_response()
    r.headers['Access-Control-Allow-Origin'] = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return r