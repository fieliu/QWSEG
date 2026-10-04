#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapted from download_modelscope(2).py; original dataset branch preserved.
Use --repo-type model for Windows-compatible SDK model downloads.

modelscope 数据集下载脚本(SDK 拿清单 + aria2c 分片下载)。

为什么不用官方 CLI:
  官方 CLI 是「单文件单线程」+ --max-workers 只增加文件并发数, 单个大文件
  (如 30G 的 ego4d zip) 只有 ~1MB/s 单连接, 太慢。
  本脚本用 modelscope SDK 拿文件清单(解决鉴权/cookie/分页), 再把每个文件
  的 resolve 直链交给 aria2c 分片下载(-x/-s 多线程), 可到几十 MB/s。

用法:
  python download_modelscope.py lmms-lab/LLaVA-OneVision-1.5-Instruct-Data \
      --local-dir /path/to/output \
      [-x 8] [-j 5] [--max-speed 30M] [--include "instruction_data/*"] [--exclude "*.zip"]

依赖: pip install modelscope aria2 (aria2c 命令)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_ENDPOINT = "https://www.modelscope.cn"
DEFAULT_REVISION = "master"
MAX_PAGE_RETRIES = 5        # 单页清单拉取失败后的重试次数
RETRY_BASE_DELAY = 3        # 退避基座(秒): 3/6/9/12/15


def fetch_files(repo_id: str, revision: str, endpoint: str,
                include, exclude, dirs=None) -> list[tuple[str, int]]:
    """用 SDK 拿文件清单。返回 [(path, size)] (仅 blob 文件)。"""
    from modelscope.hub.api import HubApi

    owner, name = repo_id.split("/", 1)
    # timeout=300: SDK 默认读超时 60s, 大仓库(数百文件)常常超过, 拉长避免误报超时
    api = HubApi(timeout=300)
    api.endpoint = endpoint

    # 拿 hub_id 也可能超时, 同样退避重试
    hub_id = None
    for attempt in range(1, MAX_PAGE_RETRIES + 1):
        try:
            hub_id, _ = api.get_dataset_id_and_type(
                dataset_name=name, namespace=owner, endpoint=endpoint)
            break
        except Exception as e:
            wait = RETRY_BASE_DELAY * attempt
            print(f"[warn] 获取 hub_id 失败(第{attempt}/{MAX_PAGE_RETRIES}次): {e}; "
                  f"{wait}s 后重试", file=sys.stderr)
            if attempt < MAX_PAGE_RETRIES:
                time.sleep(wait)
    if hub_id is None:
        print(f"[error] 获取 hub_id 连续 {MAX_PAGE_RETRIES} 次失败, 请稍后重跑。", file=sys.stderr)
        sys.exit(1)

    repo_files = []
    page_number, page_size = 1, 150
    while True:
        # 单页重试: 网络抖动/Read timed out 很常见, 退避重试几次再判定失败
        files = None
        for attempt in range(1, MAX_PAGE_RETRIES + 1):
            try:
                files = api.get_dataset_files(
                    repo_id=repo_id, revision=revision, root_path="/",
                    recursive=True, page_number=page_number, page_size=page_size,
                    dataset_hub_id=hub_id, endpoint=endpoint)
                break
            except Exception as e:
                wait = RETRY_BASE_DELAY * attempt
                print(f"[warn] 拉清单失败(第{page_number}页, 第{attempt}/{MAX_PAGE_RETRIES}次): "
                      f"{e}; {wait}s 后重试", file=sys.stderr)
                if attempt < MAX_PAGE_RETRIES:
                    time.sleep(wait)
        if files is None:
            # 重试耗尽仍失败: 不能静默用不完整清单继续下载, 否则会漏文件且无感知
            print(f"[error] 第{page_number}页清单连续 {MAX_PAGE_RETRIES} 次拉取失败, 已放弃。"
                  f"请稍后重跑本脚本(已下载文件会自动跳过、断点续传)。", file=sys.stderr)
            sys.exit(1)
        repo_files.extend(files)
        if len(files) < page_size:
            break
        page_number += 1

    out = []
    for f in repo_files:
        if f.get("Type") != "blob":
            continue
        p = f.get("Path")
        if not p:
            continue
        if dirs and not any(p == d or p.startswith(d + "/") for d in dirs):
            # --dirs: 精确下载指定目录(前缀匹配, 含嵌套子目录)
            continue
        if include and not any(_glob(p, pat) for pat in include):
            continue
        if exclude and any(_glob(p, pat) for pat in exclude):
            continue
        out.append((p, int(f.get("Size", 0) or 0)))
    return out


def _glob(path: str, pat: str) -> bool:
    import fnmatch
    return fnmatch.fnmatch(path, pat)


def make_url(repo_id: str, revision: str, endpoint: str, path: str) -> str:
    owner, name = repo_id.split("/", 1)
    # 用 resolve 直链(标准 https, 支持 Range, aria2c 分片友好)
    return f"{endpoint}/datasets/{owner}/{name}/resolve/{revision}/{path}"


def fresh_cdn_url(resolve_url: str, max_retry: int = 3) -> str | None:
    """跟随 resolve URL 的 302 跳到带时效 auth_key 的 CDN 真实地址。

    modelscope resolve 直链返回 302 → cdn-lfs-*. 含 auth_key 的 URL; auth_key 有时效,
    aria2c 若复用旧签名(断点续传)会 0 字节。所以每次给 aria2c 的最新 CDN URL。
    返回 None 表示失败(可被上层跳过本次)。
    """
    import urllib.request
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None  # 不自动跟, 我们要 Headers 里的 Location
    opener = urllib.request.build_opener(_NoRedirect)
    for attempt in range(1, max_retry + 1):
        try:
            req = urllib.request.Request(resolve_url, method="HEAD",
                                         headers={"User-Agent": "aria2/1.36"})
            try:
                opener.open(req, timeout=25)  # 极少数直接 200/206
                return resolve_url             # 没 302, 直接用原 URL
            except urllib.error.HTTPError as e:
                if e.code in (301, 302, 303, 307, 308):
                    loc = e.headers.get("Location")
                    if loc:
                        return loc.strip()
                # 其它错(404/503/...), 重试
        except Exception:
            pass
        if attempt < max_retry:
            time.sleep(2 * attempt)
    return None



def download_model(args):
    """Model branch added to the original dataset downloader; works on Windows.

    Uses the SDK for model authentication, file integrity and download recovery.
    Original aria2/wget options apply only to the unchanged dataset branch.
    """
    from modelscope.hub.api import HubApi
    from modelscope.hub.snapshot_download import snapshot_download

    local_dir = Path(args.local_dir).resolve()
    api = HubApi(endpoint=args.endpoint)
    # SDK 1.40 compatibility helpers omit SHA256; retain the raw model metadata
    # so downloaded weights can be independently checked against the repository.
    response = api.session.get(
        args.endpoint.rstrip("/") + "/api/v1/models/" + args.repo_id + "/repo/files",
        params={"Revision": args.revision, "Recursive": "true"}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("Success"):
        raise RuntimeError("Cannot list model files: " + str(payload.get("Message")))
    entries = payload["Data"]["Files"]
    selected = []
    for item in entries:
        if item.get("Type") != "blob":
            continue
        name = item.get("Path")
        if not name:
            continue
        target = (local_dir / name).resolve()
        if not target.is_relative_to(local_dir):
            raise ValueError("Unsafe repository path: " + name)
        if args.dirs and not any(name == d or name.startswith(d.rstrip("/") + "/") for d in args.dirs):
            continue
        if args.include and not any(_glob(name, p) for p in args.include):
            continue
        if args.exclude and any(_glob(name, p) for p in args.exclude):
            continue
        selected.append(item)
    if not selected:
        raise SystemExit("No model files matched; nothing downloaded")
    print("Model repository:", args.repo_id)
    print("Destination:", local_dir)
    print("Model downloads use the SDK; aria2/wget options are dataset-only.")
    for item in selected:
        print(item["Path"], item.get("Size", 0))
    if args.dry_run:
        return
    downloaded = snapshot_download(
        model_id=args.repo_id, revision=args.revision,
        local_dir=str(local_dir),
        allow_patterns=[item["Path"] for item in selected],
        max_workers=args.j, endpoint=args.endpoint)
    # Check each selected file independently, rather than aggregate byte counts.
    import hashlib
    for item in selected:
        name = item["Path"]
        target = local_dir / name
        if not target.is_file():
            raise RuntimeError("Missing downloaded file: " + name)
        expected_size = item.get("Size")
        if expected_size is not None and target.stat().st_size != int(expected_size):
            raise RuntimeError("Size mismatch: " + name)
        expected_hash = item.get("Sha256")
        if expected_hash:
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest().lower() != expected_hash.lower():
                raise RuntimeError("SHA256 mismatch: " + name)
    print("Download verified:", downloaded)


def main():
    ap = argparse.ArgumentParser(description="modelscope 数据集下载(SDK清单+aria2c分片)")
    ap.add_argument("repo_id", help="如 lmms-lab/LLaVA-OneVision-1.5-Instruct-Data")
    ap.add_argument("--local-dir", required=True, help="输出目录")
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("-s", type=int, default=None, dest="split",
                    help="分片数(aria2c -s, 无上限, 单文件切成几片并行下; 默认跟随 -x)")
    ap.add_argument("-x", type=int, default=16, dest="max_conn",
                    help="每服务器最大连接数(aria2c -x, 上限16)")
    ap.add_argument("-k", default="20M", dest="min_split_size",
                    help="最小分片大小(aria2c -k, 如 20M/10M/1M)")
    ap.add_argument("-j", type=int, default=5, help="并发下载文件数(aria2c -j)")
    ap.add_argument("--max-speed", default=None, help="限速, 如 30M / 10M(aria2c --max-download-limit)")
    ap.add_argument("--include", nargs="*", default=[], help="只下匹配的文件, 支持通配符")
    ap.add_argument("--exclude", nargs="*", default=[], help="排除匹配的文件")
    ap.add_argument("--dirs", nargs="*", default=[], help="只下指定目录(含其下所有文件), "
                    "如 --dirs 30_60_s_youtube_v0_1 1_2_m_youtube_v0_1")
    ap.add_argument("--show-files", action="store_true",
                    help="显示每个子文件的下载进度条(默认关闭, 只显示总进度)")
    ap.add_argument("--dry-run", action="store_true", help="只列清单不下载")
    ap.add_argument("--tool", default="aria2c", choices=["aria2c", "wget"],
                    help="下载工具: aria2c=分片多线程(默认, modelscope CDN 多用才快); "
                         "wget=单文件单连接顺序下(慢但稳, 部分被限流场景反而比 aria2c 强)")
    ap.add_argument("--repo-type", choices=["dataset", "model"], default="dataset",
                    help="Repository type; use model for DINOv3 weights")
    args = ap.parse_args()
    if args.repo_type == "model":
        download_model(args)
        return

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"📋 拉取文件清单: {args.repo_id} (revision={args.revision})")
    files = fetch_files(args.repo_id, args.revision, args.endpoint,
                        args.include, args.exclude, dirs=args.dirs)
    total_size = sum(s for _, s in files)
    print(f"📊 共 {len(files)} 个文件, 总大小 {total_size/1e9:.2f} GB")

    if not files:
        print("无文件(检查 repo_id / revision / include-exclude 过滤)")
        return

    # 断点续传: 排除“已完成”的
    # 判定完成 = 大小一致(>0) **且** 旁边没有 aria2 控制文件(*.aria2)。
    # 注意: aria2 分片/预分配会让大小先到顶但数据未齐, 此时一定会有 .aria2 残留,
    #       只看大小会把残档误判为完成, 必须同时要求 .aria2 不存在。
    todo = []
    n_stale = 0   # 大小已到但带 .aria2 的真残档(旧逻辑会漏掉)
    for path, size in files:
        dest = local_dir / path
        af = Path(str(dest) + ".aria2")
        if dest.exists() and size > 0 and dest.stat().st_size == size and not af.exists():
            continue
        if af.exists():
            n_stale += 1
        todo.append((path, size))
    print(f"  待下载 {len(todo)} 个(跳过已完成的 {len(files)-len(todo)} 个; "
          f"其中带 .aria2 残档 {n_stale} 个将断点续传补齐)")

    if not todo:
        print("全部已下载完成。")
        return

    if args.dry_run:
        print("\n前 20 个待下载文件:")
        for p, s in todo[:20]:
            print(f"  {s/1e6:9.1f}MB  {p}")
        return

    # 生成 aria2c 输入文件: "URL\n dir=...\n out=..."
    # 关键: 每个文件先 resolve → 当秒的含 auth_key CDN URL, 再交给 aria2,
    #        否则 aria2 复用旧签名会 0 字节卡死
    list_file = local_dir / ".aria2c_input.txt"
    dest_map = {}   # 文件路径 -> (目标绝对路径, 期望大小)
    lines = []
    n_url_fail = 0
    print(f"🔑 正在批量解析 {len(todo)} 个文件的 CDN 直链(含 auth_key)...")
    for idx, (path, size) in enumerate(todo, 1):
        resolve = make_url(args.repo_id, args.revision, args.endpoint, path)
        cdn = fresh_cdn_url(resolve)
        if not cdn:
            n_url_fail += 1
            print(f"   [{idx}/{len(todo)}] 取 CDN 失败, 暂跳: {path}")
            continue
        dest = (local_dir / path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        lines.append(cdn)
        lines.append(f" dir={dest.parent}")
        lines.append(f" out={dest.name}")
        lines.append(" continue=true")
        dest_map[path] = (dest, size)
        if idx % 10 == 0 or idx == len(todo):
            print(f"   已解析 {idx}/{len(todo)}  (失败 {n_url_fail})")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not dest_map:
        print("❌ 所有 CDN 链接都解析失败; 请检查网络/鉴权后重试")
        sys.exit(1)
    if n_url_fail:
        print(f"⚠️  {n_url_fail} 个文件因 CDN URL 取不到被暂时跳过; 可重跑脚本继续补")

    # 总需下载字节: 完整文件视为 0, )残档/缺失 = size; 这样进度不被“预占满 size”
    # 的残档虚充到 ~100%. 先占位, 下方 _true_total_download() 会覆写 dest_map 建立正确值.
    total_download = 0
    def _current_downloaded():
        """真实已下字节(只数真正落盘的物理块, 不数 ftruncate 预置的全尺寸占位).

        关键: aria2 --file-allocation=none 在 NFS 上退化为 ftruncate, 启动瞬间
        st_size 就涨到全尺寸(4.9GB), 但其中是 sparse 空洞 —— st_size 完全不
        能用来衡量"已下载多少"。st_blocks*512 才是真实占有磁盘的字节。
          - 有 .aria2(进行中/残档): 用 st_blocks*512 (真实已写盘的字节, 空洞=0)
          - 无 .aria2(已完成)      : 用 st_size (文件就是满的)
        """
        got = 0
        for path, (dest, _) in dest_map.items():
            af = Path(str(dest) + ".aria2")
            if not dest.exists():
                continue
            st = dest.stat()
            if af.exists():
                got += st.st_blocks * 512   # 真实落盘, 不含sparse空洞/预置占位
            else:
                got += st.st_size
        return got

    # total_download(进度分母) = 本次要处理的全部文件的目标尺寸总和(含已完成与残档).
    # 分子 _current_downloaded() 用 st_blocks 真实落盘; 两者口径一致:
    #   已完成的算 st_size(==目标) → 该文件直接计满
    #   残档/缺失的算 st_blocks(真实已下) → 随落盘平滑涨, 不再"启动即96%"
    # 完成后 分子==分母 → 100.0%
    def _true_total_download():
        return sum(size for _, (dest, size) in dest_map.items())

    total_download = _true_total_download()

    start_bytes = _current_downloaded()

    log_level = "notice" if args.show_files else "warn"   # 默认只显示警告, 不刷屏子文件

    # ---------- wget 分支: 单文件顺序下, --continue 断点续传 ----------
    # 适合单连接被限流的镜像站 (尤其是 xet-bridge 那种签名+并发标记限流)。
    if args.tool == "wget":
        import shutil
        if not shutil.which("wget"):
            print("❌ aria2c 不可用, wget 也未安装, 无法下载"); sys.exit(1)
        print(f"\n🚀 开始 wget 下载 (单文件顺序下, 断点续传)  Ctrl+C 中断, 重跑自动续)")
        for path, (dest, size) in dest_map.items():
            af = Path(str(dest) + ".aria2")
            # 若 dest 已存在 + 有 .aria2 = 有 sparse 漏数据; wget -c 不能补洞 (416),
            # 必须先删 dest 从头下(与 hfd.sh aria2c 分支同 ide)
            if dest.exists() and af.exists():
                try:
                    dest.unlink()
                    af.unlink(missing_ok=True)
                except Exception as e:
                    print(f"  ! 清残留失败 {path}: {e}")
            if not dest.exists() or af.exists() \
               or dest.stat().st_size < size:
                # 需要续传/下载的整个文件: 先重新取当秒有效 CDN URL
                resolve = make_url(args.repo_id, args.revision, args.endpoint, path)
                cdn = fresh_cdn_url(resolve)
                if not cdn:
                    print(f"  ! 取 CDN 失败, 暂跳: {path}"); continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"  → wget: {path}  ({size/1e9:.2f} GB)")
                wcmd = ["wget", "-c", "--progress=bar", "-O", str(dest), cdn]
                if args.max_speed:
                    wcmd.insert(1, "--limit-rate=" + args.max_speed.lower().replace("m", "m"))
                subprocess.run(wcmd)
        # 收尾校验
        n_still = sum(1 for _, (dest, _) in dest_map.items() if not dest.exists())
        leftover = sum(1 for _, (dest, _) in dest_map.items()
                       if Path(str(dest) + ".aria2").exists())
        print(f"\n===== wget 下载汇总 =====")
        print(f"  缺文件: {n_still}  带 .aria2 残档: {leftover}")
        if n_still == 0 and leftover == 0:
            print("✅ 全部完成")
        else:
            print("⚠️  还有未完成, 重跑 continuer")
        return

    # ---------- aria2c 分支 (默认, 多连接分片) ----------
    split = args.split if args.split else args.max_conn
    cmd = [
        "aria2c", f"--console-log-level={log_level}", "--summary-interval=1",
        "--file-allocation=none",
        "-x", str(min(args.max_conn, 16)), "-j", str(args.j), "-s", str(split),
        "-k", args.min_split_size, "-c",       # 断点续传
        "--connect-timeout=120", "--timeout=600", "--max-tries=0", "--retry-wait=5",
        f"--min-split-size={args.min_split_size}",
    ]
    if args.max_speed:
        cmd += ["--max-download-limit", args.max_speed]
    cmd += ["-i", str(list_file)]

    print(f"\n🚀 开始 aria2c 下载 (分片 -s{split} 单片最小 {args.min_split_size} "
          f"连接 -x{args.max_conn} 并发 -j{args.j}"
          + (f" 限速 {args.max_speed}" if args.max_speed else "")
          + "  Ctrl+C 断点续传, 重跑自动续)")

    # 后台总进度监控: 每秒统计已下载字节 -> 总进度/速度/ETA
    import threading
    proc = None
    stop = {"v": False}

    def _monitor():
        import time as _t
        last = start_bytes
        t0 = _t.time()
        while not stop["v"]:
            _t.sleep(1)
            cur = _current_downloaded()
            pct = cur / total_download * 100 if total_download else 0
            speed = (cur - last) / 1.0
            last = cur
            done_files = sum(1 for path, (dest, size) in dest_map.items()
                             if dest.exists() and size > 0
                             and dest.stat().st_size >= size
                             and not Path(str(dest) + ".aria2").exists())
            remain = total_download - cur
            eta = remain / speed if speed > 0 else 0
            sys.stdout.write(
                f"\r  ⏳ 总进度 {cur/1e9:.2f}GB/{total_download/1e9:.2f}GB ({pct:.1f}%) | "
                f"速度 {speed/1e6:.1f}MB/s | 剩余 {eta/60:.1f} 分钟 | "
                f"完成 {done_files}/{len(todo)} 文件     ")
            sys.stdout.flush()

    mon = threading.Thread(target=_monitor, daemon=True)
    mon.start()
    # aria2c 输出重定向到日志文件(默认不刷屏); --show-files 时才透传到终端
    log_path = local_dir / ".aria2c_download.log"
    if args.show_files:
        outf, errf = None, None
    else:
        outf = open(log_path, "wb")
        errf = subprocess.STDOUT
    try:
        proc = subprocess.run(cmd, stdout=outf, stderr=errf)
    finally:
        if outf is not None:
            outf.close()
        stop["v"] = True
        time.sleep(1.1)
        sys.stdout.write("\n")
        sys.stdout.flush()

    # 汇总(真实校验退出码 + 磁盘字节, 不无条件刷 ✅)
    rc = proc.returncode if proc else -1
    final = _current_downloaded()
    # 仍有残档(带 .aria2)就视为未真正完成
    leftover = sum(1 for path, (dest, _) in dest_map.items()
                   if Path(str(dest) + ".aria2").exists())
    print(f"\n===== 下载汇总 =====")
    print(f"  已下载 {final/1e9:.2f}GB / 计划 {total_download/1e9:.2f}GB "
          f"(aria2c exit={rc}, 仍带 .aria2 残档 {leftover} 个)")
    if rc == 0 and leftover == 0 and final >= total_download:
        print("✅ 真正下载完成")
    else:
        print("⚠️  未完成或有中断: 重跑本脚本会自动断点续传;"
              " 若多次卡在 0, 检查网速/限速(--max-speed) 与 modelscope 是否限流")


if __name__ == "__main__":
    main()
