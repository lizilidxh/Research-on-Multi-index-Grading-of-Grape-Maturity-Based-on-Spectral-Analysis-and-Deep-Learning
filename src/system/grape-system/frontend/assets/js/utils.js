// 步骤样式类计算
function getStepClass(currentStep, targetStep) {
  if (currentStep > targetStep) return 'step-done';
  if (currentStep === targetStep) return 'step-active';
  return 'step-wait';
}

// 步骤文字颜色计算
function getStepColor(currentStep, targetStep) {
  if (currentStep > targetStep) return '#10b981';
  if (currentStep === targetStep) return '#a78bfa';
  return '#64748b';
}

// 设置上传图片
function setUploadedImage(file, previewUrlRef, imgFileRef, imgResultRef, imgErrorRef, stepRef) {
  if (!file) return;
  imgFileRef.value = file;
  previewUrlRef.value = URL.createObjectURL(file);
  imgResultRef.value = null;
  imgErrorRef.value = '';
  stepRef.value = 1;
}

// 初始化历史记录图表
function initHistoryCharts(history, matNames, matColors, histDonutRef, histLineRef) {
  // 销毁旧图表
  if (window.hdInst) window.hdInst.destroy();
  if (window.hlInst) window.hlInst.destroy();

  // 饼图（成熟度分布）
  if (histDonutRef.value) {
    const count = [0, 0, 0, 0];
    history.forEach(item => count[item.result.label]++);

    window.hdInst = new Chart(histDonutRef.value, {
      type: 'doughnut',
      data: {
        labels: matNames,
        datasets: [{
          data: count,
          backgroundColor: matColors,
          borderWidth: 2,
          borderColor: '#111827'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '65%',
        plugins: {
          legend: {
            labels: {
              color: '#94a3b8',
              font: { size: 11 }
            }
          }
        }
      }
    });
  }

  // 折线图（置信度趋势）
  if (histLineRef.value) {
    const items = [...history].reverse();
    window.hlInst = new Chart(histLineRef.value, {
      type: 'line',
      data: {
        labels: items.map((_, i) => `#${i+1}`),
        datasets: [{
          label: '置信度 (%)',
          data: items.map(item => +(item.result.confidence * 100).toFixed(1)),
          borderColor: '#a78bfa',
          backgroundColor: 'rgba(167,139,250,.1)',
          pointBackgroundColor: items.map(item => matColors[item.result.label]),
          pointRadius: 6,
          tension: 0.3,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: '#64748b' }, grid: { color: '#1e2d45' } },
          y: { ticks: { color: '#64748b' }, grid: { color: '#1e2d45' }, min: 0, max: 100 }
        },
        plugins: {
          legend: { labels: { color: '#94a3b8' } }
        }
      }
    });
  }
}