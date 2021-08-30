TimeCounter
==

## Python 时间统计函数(支持 PyTorch CUDA)

* 无侵入的增加打印时间功能
* 支持 cuda 同步统计时间
* 支持平均时间打印
* 支持打印间隔，WarmUp 时长设置
* 支持 with 上下文，可用于某个函数内部某一段程序统计时长操作 (暂未实现)

原文地址:https://zhuanlan.zhihu.com/p/404204964

例子:
```python
if __name__ == '__main__': 
 
    @TimeCounter.count_time() 
    def fun1(): 
        time.sleep(2) 
 
    @TimeCounter.count_time() 
    def fun2(): 
        time.sleep(1) 
 
    for _ in range(20): 
        fun1() 
        for _ in range(2): 
            fun2() 
```
用法：

在任意函数上，加上 @TimeCounter.count_time() 即可，如果想定制化参数，例如可以：
```python
@TimeCounter.count_time(log_interval=10, warmup_interval=5, with_sync=True) 
```