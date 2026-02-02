#!/bin/bash
# 清除所有代理环境变量

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset ALL_PROXY
unset no_proxy
unset NO_PROXY

echo "✓ 代理环境变量已清除"
echo "当前代理设置："
env | grep -i proxy || echo "  (无代理设置)"
