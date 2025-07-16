optimum-cli export onnx --model qwen2.5-1.5B-merge --task text-generation-with-past --fp16 --device cuda qwen2.5-1.5B_onnx/
optimum-cli export onnx --model qwen2.5-1.5B-merge1 --task text-generation-with-past --fp16 --device cuda qwen2.5-1.5B_onnx1/


## tensorrt解压后如何操作

```
ls: cannot access /root/resume_summary/TensorRT-10.0.1.6/targets/x86_64-linux-gnu/lib:: No such file or directory
```

是因为你的 `LD_LIBRARY_PATH` 环境变量 **末尾多了一个冒号 `:`**，它表示一个“空路径”，所以当你运行 `ls $LD_LIBRARY_PATH` 时，系统尝试访问这个不存在的路径就会报错。

---

## ✅ 当前问题总结

你运行了：

```bash
echo $LD_LIBRARY_PATH
```

输出是：

```
/root/resume_summary/TensorRT-10.0.1.6/targets/x86_64-linux-gnu/lib:
```

注意结尾的冒号 `:` —— 它是问题所在！

---

## ✅ 解决方案：清空并重新设置正确的环境变量

### 🔧 步骤一：清除当前错误的路径

```bash
unset LD_LIBRARY_PATH
```

### ✅ 步骤二：正确设置环境变量（确保没有多余的冒号）

```bash
export LD_LIBRARY_PATH=$(pwd)/TensorRT-10.0.1.6/targets/x86_64-linux-gnu/lib
```

> ⚠️ 注意：这里我们不加 `:$LD_LIBRARY_PATH`，因为此时它是空的，加上反而会多出一个 `:`。

---

### 🔄 步骤三：更新动态链接缓存

```bash
sudo ldconfig
```

---

## 📌 验证是否成功

### ✅ 查看当前 `LD_LIBRARY_PATH` 值：

```bash
echo $LD_LIBRARY_PATH
```

你应该看到类似输出（**结尾没有冒号**）：

```
/root/resume_summary/TensorRT-10.0.1.6/targets/x86_64-linux-gnu/lib
```

✅ 成功！

---

### ✅ 查看 TensorRT 动态库是否存在：

```bash
ls $LD_LIBRARY_PATH
```

你应该能看到如下文件（表示 TensorRT 的库已经正确解压）：

```
libnvinfer.so.10         libnvinfer_plugin.so.10   libnvonnxparser.so.10    libnvparsers.so.10
```

---

## ✅ 永久保存路径（推荐）

将环境变量写入 `~/.bashrc` 文件中，这样每次打开终端时都会自动加载：

```bash
echo 'export LD_LIBRARY_PATH=$(pwd)/TensorRT-10.0.1.6/targets/x86_64-linux-gnu/lib' >> ~/.bashrc
source ~/.bashrc
```

```aiignore
cd TensorRT-10.0.1.6/python/
ls
pip install tensorrt-10.0.1-cp310-none-linux_x86_64.whl
```

