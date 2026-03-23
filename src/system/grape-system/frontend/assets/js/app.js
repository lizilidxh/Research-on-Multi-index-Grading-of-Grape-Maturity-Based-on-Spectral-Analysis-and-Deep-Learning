const { createApp, ref, computed, onMounted, watch, nextTick } = Vue;

createApp({
  setup() {
    // 基础状态
    const page = ref('detect');
    const apiBase = ref('http://localhost:5000');
    const connected = ref(false);
    const connLoading = ref(false);
    const sysStats = ref({});

    // 连接状态计算属性
    const connClass = computed(() => ({
      'conn-ok': connected.value,
      'conn-err': !connected.value && !connLoading.value,
      'conn-loading': connLoading.value,
    }));

    const connText = computed(() =>
      connLoading.value ? '连接中…' : connected.value ? '后端已连接' : '未连接'
    );

    // 检查连接（封装API调用）
    async function checkConn() {
      connLoading.value = true;
      try {
        connected.value = await checkConnection(apiBase.value);
      } catch {
        connected.value = false;
      } finally {
        connLoading.value = false;
      }
    }

    // 加载系统统计
    async function loadStats() {
      sysStats.value = await loadSystemStats(apiBase.value);
    }

    // 图像上传相关状态
    const imgFile = ref(null);
    const previewUrl = ref('');
    const dragging = ref(false);
    const predicting = ref(false);
    const imgError = ref('');
    const imgResult = ref(null);
    const step = ref(1);

    // 历史记录
    const history = ref([]);
    const histDonut = ref(null);
    const histLine = ref(null);

    // 步骤样式方法（封装工具函数）
    const stepClass = (s) => getStepClass(step.value, s);
    const stepColor = (s) => getStepColor(step.value, s);

    // 图片处理方法
    const onImgChange = (e) => setUploadedImage(e.target.files[0], previewUrl, imgFile, imgResult, imgError, step);
    const onDrop = (e) => {
      dragging.value = false;
      setUploadedImage(e.dataTransfer.files[0], previewUrl, imgFile, imgResult, imgError, step);
    };

    const resetImg = () => {
      imgFile.value = null;
      previewUrl.value = '';
      imgResult.value = null;
      imgError.value = '';
      step.value = 1;
    };

    // 提交图片识别
    async function submitImg() {
      if (!imgFile.value) return;
      predicting.value = true;
      imgError.value = '';
      step.value = 2;

      try {
        const result = await submitImageForPrediction(apiBase.value, imgFile.value);
        if (result.error) {
          imgError.value = result.error;
          step.value = 1;
        } else {
          imgResult.value = result;
          step.value = 3;
          // 添加到历史记录
          history.value.unshift({
            previewUrl: previewUrl.value,
            result: result,
            time: new Date().toLocaleTimeString(),
          });
        }
      } catch (e) {
        imgError.value = '请求失败，请确认后端已启动：' + e.message;
        step.value = 1;
      } finally {
        predicting.value = false;
      }
    }

    // 查看历史记录
    function viewHistory(item) {
      previewUrl.value = item.previewUrl;
      imgFile.value = { name: '(历史记录)', size: 0 };
      imgResult.value = item.result;
      step.value = 3;
      page.value = 'detect';
    }

    // 监听历史页面切换，初始化图表
    watch(page, async (val) => {
      if (val !== 'history' || history.value.length < 2) return;
      await nextTick();
      initHistoryCharts(history.value, matNames, matColors, histDonut, histLine);
    });

    // 初始化
    onMounted(async () => {
      await checkConn();
      await loadStats();
    });

    // 返回模板使用的变量和方法
    return {
      // 基础状态
      page, apiBase, connected, connLoading, connClass, connText, sysStats,
      // 配置常量
      navItems, matNames, matColors, imgPipeline, matDescriptions, colorFeatures,
      // 图像上传
      imgFile, previewUrl, dragging, predicting, imgError, imgResult, step,
      // 历史记录
      history, histDonut, histLine,
      // 方法
      checkConn, onImgChange, onDrop, resetImg, submitImg, viewHistory,
      stepClass, stepColor
    };
  }
}).mount('#app');