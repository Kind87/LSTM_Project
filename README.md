# LSTM 皮革瑕疵檢測

比較 CNN＋LSTM、精簡版 PatchCore、Isolation Forest 和局部特徵版 CutPaste 在三種亮度下的皮革瑕疵判定。網站使用 React、Tailwind CSS 和 FastAPI，支援上傳影像、切換光照、比較分數及查看異常熱圖。

- [四種方法的實驗結果](lstm-day1/four-model-results/metrics.csv)
- `lstm-day1/` 放完整的訓練與評估程式、依賴、模型權重及結果。
- `leather-ui/` 放網站的 React 前端、FastAPI 後端及六張示範圖。

## 啟動網站

準備 Node.js 22.12 或更新版本、npm 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。在 Windows 從專案根目錄執行 `leather-ui\start.cmd`，然後打開 [本機網站](http://127.0.0.1:8765/)。首次啟動需要網路，以安裝依賴並取得 ResNet-18 權重。網站使用已附的模型權重和六張內建樣本。可上傳自己的 JPG、PNG 或 WebP 圖片，每張最多 10 MB。

如需換埠，執行 `leather-ui\start.cmd 8766`。修改前端後，執行 `leather-ui\start.cmd 8765 rebuild` 重新建置。

## 重跑作業實驗

從 [MVTec AD 官方資料集](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) 取得完整 leather 類別。將它放在專案根目錄的 `mvtec-ad/leather`，其下應有 `train`、`test` 與 `ground_truth`。可以把下方 `$leatherData` 改成你的資料集所在位置。

```powershell
$leatherData = './mvtec-ad/leather'
uv sync --locked --project ./lstm-day1 --python 3.12
uv run --locked --project ./lstm-day1 python ./lstm-day1/run_experiment.py --data-dir $leatherData --output-dir ./outputs/preprocess --device cpu
uv run --locked --project ./lstm-day1 python ./lstm-day1/run_full_experiment.py --data-dir $leatherData --output-dir ./outputs/baseline --cache-dir ./cache --device cpu
uv run --locked --project ./lstm-day1 python ./lstm-day1/add_anomaly_methods.py --data-dir $leatherData --baseline-results ./outputs/baseline --output-dir ./outputs/four-models
```

重跑產物寫到 `outputs/`。其中 `four-models/metrics.csv` 含 12 組結果，`image_scores.csv` 含 1,488 筆逐圖判定。

## 影像來源

網站的六張示範圖來自 MVTec AD 的 leather 官方測試集，提供者為 MVTec Software GmbH。報告與比較圖若使用該資料的影像，亮度調整、裁切、標記及熱圖為本作業的視覺化處理。這些影像依資料集原本的 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hant) 授權使用。
