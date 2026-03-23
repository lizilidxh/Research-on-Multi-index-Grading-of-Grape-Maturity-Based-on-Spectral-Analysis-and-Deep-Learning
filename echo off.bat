@echo off
chcp 65001 > nul
echo ==============================================
echo          葡萄成熟度识别项目 - 文件夹生成工具
echo ==============================================
echo.

:: 设置项目根目录（默认是bat文件所在目录）
set "PROJECT_ROOT=%~dp0"
echo 项目根目录：%PROJECT_ROOT%
echo.

:: 1. 创建核心文件夹
echo [1/2] 正在创建基础文件夹...
mkdir "%PROJECT_ROOT%data" 2>nul
mkdir "%PROJECT_ROOT%src" 2>nul
mkdir "%PROJECT_ROOT%results" 2>nul
mkdir "%PROJECT_ROOT%results\聚类结果" 2>nul
mkdir "%PROJECT_ROOT%results\模型结果" 2>nul
mkdir "%PROJECT_ROOT%results\可视化图" 2>nul

:: 2. 创建src文件夹下的空py文件（带基础注释）
echo [2/2] 正在创建源代码文件...
:: 01_聚类标签生成.py
echo ^# -*- coding: utf-8 -*- > "%PROJECT_ROOT%src\01_聚类标签生成.py"
echo ^""" >> "%PROJECT_ROOT%src\01_聚类标签生成.py"
echo 第一步：葡萄成熟度多指标聚类标签生成 >> "%PROJECT_ROOT%src\01_聚类标签生成.py"
echo 功能：读取光谱数据，用K-means聚类生成4个成熟度标签 >> "%PROJECT_ROOT%src\01_聚类标签生成.py"
echo ^""" >> "%PROJECT_ROOT%src\01_聚类标签生成.py"

:: 02_光谱预处理.py
echo ^# -*- coding: utf-8 -*- > "%PROJECT_ROOT%src\02_光谱预处理.py"
echo ^""" >> "%PROJECT_ROOT%src\02_光谱预处理.py"
echo 第二步：光谱数据预处理 >> "%PROJECT_ROOT%src\02_光谱预处理.py"
echo 功能：SG平滑、MSC、SNV等预处理，去除噪声和基线漂移 >> "%PROJECT_ROOT%src\02_光谱预处理.py"
echo ^""" >> "%PROJECT_ROOT%src\02_光谱预处理.py"

:: 03_预测模型构建.py
echo ^# -*- coding: utf-8 -*- > "%PROJECT_ROOT%src\03_预测模型构建.py"
echo ^""" >> "%PROJECT_ROOT%src\03_预测模型构建.py"
echo 第三步：预测模型构建 >> "%PROJECT_ROOT%src\03_预测模型构建.py"
echo 功能：构建SVM、CNN、BP神经网络等基础预测模型 >> "%PROJECT_ROOT%src\03_预测模型构建.py"
echo ^""" >> "%PROJECT_ROOT%src\03_预测模型构建.py"

:: 04_模型优化.py
echo ^# -*- coding: utf-8 -*- > "%PROJECT_ROOT%src\04_模型优化.py"
echo ^""" >> "%PROJECT_ROOT%src\04_模型优化.py"
echo 第四步：模型优化与性能提升 >> "%PROJECT_ROOT%src\04_模型优化.py"
echo 功能：优化模型结构，引入正则化/集成学习，提升泛化能力 >> "%PROJECT_ROOT%src\04_模型优化.py"
echo ^""" >> "%PROJECT_ROOT%src\04_模型优化.py"

:: 05_界面系统开发.py
echo ^# -*- coding: utf-8 -*- > "%PROJECT_ROOT%src\05_界面系统开发.py"
echo ^""" >> "%PROJECT_ROOT%src\05_界面系统开发.py"
echo 第五步：葡萄成熟度智能识别系统开发 >> "%PROJECT_ROOT%src\05_界面系统开发.py"
echo 功能：基于PyQt5开发PC端桌面应用，集成数据采集、预处理、预测等功能 >> "%PROJECT_ROOT%src\05_界面系统开发.py"
echo ^""" >> "%PROJECT_ROOT%src\05_界面系统开发.py"

:: 3. 创建数据说明文件
echo 请将高光谱数据文件（如Excel/CSV）放入此文件夹 > "%PROJECT_ROOT%data\数据说明.txt"
echo 支持格式：.xlsx .csv .txt >> "%PROJECT_ROOT%data\数据说明.txt"

echo.
echo ==============================================
echo ✅ 文件夹结构创建完成！
echo 📂 生成的结构：
echo   你的项目名/
echo   ├── data/                （放Excel/CSV数据）
echo   ├── src/                 （源代码文件夹）
echo   │   ├── 01_聚类标签生成.py
echo   │   ├── 02_光谱预处理.py
echo   │   ├── 03_预测模型构建.py
echo   │   ├── 04_模型优化.py
echo   │   └── 05_界面系统开发.py
echo   └── results/             （结果文件夹）
echo       ├── 聚类结果/
echo       ├── 模型结果/
echo       └── 可视化图/
echo ==============================================
pause