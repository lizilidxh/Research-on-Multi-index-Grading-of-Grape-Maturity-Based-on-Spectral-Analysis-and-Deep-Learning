// 检查后端连接状态
async function checkConnection(apiBase) {
  try {
    const response = await fetch(`${apiBase}/api/health`);
    const data = await response.json();
    return !!(data.img_model || data.spec_model);
  } catch (error) {
    return false;
  }
}

// 加载系统统计数据
async function loadSystemStats(apiBase) {
  try {
    const response = await fetch(`${apiBase}/api/stats`);
    return await response.json();
  } catch (error) {
    return {};
  }
}

// 提交图片进行识别
async function submitImageForPrediction(apiBase, file) {
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(`${apiBase}/api/predict/image`, {
    method: 'POST',
    body: formData
  });

  return await response.json();
}