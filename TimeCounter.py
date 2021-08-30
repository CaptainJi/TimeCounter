import time
import torch


class TimeCounter:
    count = 0
    pure_inf_time = 0
    log_interval = 1
    warmup_interval = 0
    with_sync = True
    names = dict()

    # Avoid instantiating every time
    @classmethod
    def count_time(cls, log_interval=1, warmup_interval=0, with_sync=True):
        def _register(func):
            if func.__name__ in cls.names:
                raise RuntimeError('The registered function name cannot be repeated!')
                # When adding on multiple functions, we need to ensure that the
            # data does not interfere with each other
            cls.names[func.__name__] = dict(count=0,
                                            pure_inf_time=0,
                                            log_interval=log_interval,
                                            warmup_interval=warmup_interval,
                                            with_sync=with_sync)

            def fun(*args, **kwargs):
                count = cls.names[func.__name__]['count']
                pure_inf_time = cls.names[func.__name__]['pure_inf_time']
                log_interval = cls.names[func.__name__]['log_interval']
                warmup_interval = cls.names[func.__name__]['warmup_interval']
                with_sync = cls.names[func.__name__]['with_sync']

                count += 1
                cls.names[func.__name__]['count'] = count

                if with_sync and torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                result = func(*args, **kwargs)

                if with_sync and torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start_time

                if count >= warmup_interval:
                    pure_inf_time += elapsed
                    cls.names[func.__name__]['pure_inf_time'] = pure_inf_time

                    if count % log_interval == 0:
                        times_per_count = 1000 * pure_inf_time / count
                        # print(f'[{func.__name__}]-{count} times per count: {times_per_count:.1f} ms', flush=True)
                        print(f'[{func.__name__}]第{count}次运行耗时: {times_per_count:.1f} ms', flush=True)

                return result

            return fun

        return _register

# example:
if __name__ == '__main__':

    @TimeCounter.count_time()
    def fun1():
        time.sleep(2)


    @TimeCounter.count_time(log_interval=10, warmup_interval=5, with_sync=True)
    def fun2():
        time.sleep(1)


    for _ in range(20):
        fun1()
        for _ in range(2):
            fun2()