"""稳健性核查（交付附证）：为什么没有采用「波动率状态」信号。

探索中曾发现 `pred = 训练段波动率中位数 - 当期 vol60` 这一因果信号能在
2024-07~2025-12 样本外同时通过精度账与交易账。但它是**看着样本外结果挑出来**的，
本脚本用 walk-forward（每段只用该段之前的数据定阈值）把它放到多个时段检验，
证明它只在最后一段牛市有效，属于 `测试集参与调参与模型选择`（有缺陷）的反例，
故不作为交付模型。

同时输出训练段/样本外的市场状态对照，说明本任务判负的结构性原因。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATA = (r"D:/awork/akf/llmagent/cankao/gupiao/"
        r"计科3班20232131005魏煜桓金融数据分析与智能量化交易应用课程期末设计/code/data/601318.csv")
H = 5
COST = 0.0005
WINDOWS = [
    ("2022-07-01", "2023-06-30"),
    ("2023-01-01", "2024-06-28"),
    ("2024-07-01", "2025-12-31"),
]


def load():
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df


def positions(signal_idx, horizon, n):
    pos = np.zeros(n)
    nf = 0
    for idx in np.sort(np.asarray(signal_idx, dtype=int)):
        e = idx + 1
        if e < nf or e >= n:
            continue
        x = min(e + horizon, n)
        pos[e:x] = 1.0
        nf = x
    return pos


def stats(close, pos, cost, start):
    ret = np.zeros(len(close))
    ret[1:] = close[1:] / close[:-1] - 1
    ch = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = (pos * ret - ch * cost)[start:]
    eq = np.cumprod(1 + net)
    cum = eq[-1] - 1
    sh = net.mean() / net.std(ddof=1) * np.sqrt(252) if net.std(ddof=1) > 0 else float("nan")
    return cum, sh


def walk_forward(df, pred, start_date, end_date, name):
    close = df["close"].to_numpy(float)
    label = (df["close"].shift(-H) / df["close"] - 1).to_numpy()
    dates = df["date"]
    te = ((dates >= start_date) & (dates <= end_date)).to_numpy()
    pre = (dates < start_date).to_numpy() & ~np.isnan(pred) & ~np.isnan(label)
    thr = np.median(pred[pre]) if pre.sum() >= 60 else np.median(pred[te & ~np.isnan(pred)])
    idx = np.where(te & (pred > thr) & ~np.isnan(pred))[0]
    pos = positions(idx, H, len(close))
    start = int(np.argmax(te))
    cum, sh = stats(close, pos, COST, start)
    b_cum, b_sh = stats(close, np.ones(len(close)), COST, start)
    truth = label[te]
    ok = ~np.isnan(truth) & ~np.isnan(pred[te])
    acc = float(np.mean(np.sign(pred[te][ok]) == np.sign(truth[ok])))
    print(f"  {name:10s} [{start_date}~{end_date}] cum {100*cum:+7.2f}% sh {sh:5.2f} "
          f"acc {acc:.3f} expo {pos[start:].mean():.2f} | 买入持有 {100*b_cum:+7.2f}% sh {b_sh:5.2f}")


if __name__ == "__main__":
    df = load()
    S = df["close"].astype(float)
    y = S.shift(-H) / S - 1
    cut = pd.Timestamp("2024-07-01")
    tr = (df["date"] < cut).to_numpy()
    te = ~tr

    print("== 市场状态对照 ==")
    for tag, m in (("训练段 2022-01~2024-06", tr), ("样本外 2024-07~2025-12", te)):
        v = y[m].dropna()
        print(f"  {tag}: 5日收益均值 {v.mean():+.4%}，上涨占比 {(v > 0).mean():.3f}，"
              f"零预测 RMSE {np.sqrt((v ** 2).mean()):.4%}，常数均值 RMSE {v.std():.4%}")

    print("\n== 训练段趋势状态的未来 5 日收益（说明为何趋势模型在样本外偏空） ==")
    for n in (20, 60):
        st = S > S.rolling(n).mean()
        for s in (True, False):
            m = tr & (st == s).to_numpy() & y.notna().to_numpy()
            print(f"  close>MA{n} = {str(s):5s}: 训练段 n={m.sum():3d}，"
                  f"均值 {y[m].mean():+.4%}，上涨占比 {(y[m] > 0).mean():.3f}")

    print("\n== 被否决的「波动率状态」信号 walk-forward（阈值只用该段之前的数据） ==")
    r1 = np.log(S).diff(1)
    for w in (20, 60, 120):
        v = r1.rolling(w).std()
        med_past = v.shift(1).expanding(min_periods=60).median()   # 因果：只用过去
        pred = (med_past - v).to_numpy(float)
        print(f"  -- vol{w} --")
        for a, b in WINDOWS:
            walk_forward(df, pred, pd.Timestamp(a), pd.Timestamp(b), f"med-vol{w}")
