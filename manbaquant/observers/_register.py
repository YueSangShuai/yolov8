from typing import Union
# from hmquant.utils.config import filter_args_of_cfg
from .observer_abc import ObserverABC

class BaseRegister:
    """Register to map a str and a obj"""

    def __init__(
        self, D: dict = None, key_type=None, value_type=None, add_str_name=True
    ):
        """用来从一个key来映射一个Class的注册器

        Args:
            D (dict, optional): 用来保存映射关系的字典. Defaults to None.
            key_type (_type_, optional): 限定key的类型. Defaults to None.
            value_type (_type_, optional): 限定value的类型. Defaults to None.
            add_str_name (bool, optional): 在注册的时候是否自动将Class的名字也作为一个key. Defaults to True.
        """
        if D is None:
            D = dict()
        self.D = D
        self.key_type = key_type
        self.value_type = value_type
        self.add_str_name = add_str_name

    def __repr__(self):
        return self.D.__repr__()

    @property
    def keys(self):
        return self.D.keys()

    @property
    def values(self):
        return self.D.values()

    def add(self, *keys, **info):
        """向注册器中增加一个映射关系

        Args:
            domain (str, optional): 如果domain不是None,说明这个要在某个domain中进行查表. Defaults to None.
            keys: 任意数量的列表,用来逐个生成映射对的key
        """
        domain = info.pop("domain", None)

        def insert(value):
            if self.value_type:
                assert isinstance(value, self.value_type) or issubclass(
                    value, self.value_type
                ), "must matching"
            nonlocal keys, info, domain
            keys = set(keys)
            if hasattr(value, "__name__") and self.add_str_name:
                keys.add(value.__name__)

            for key in keys:
                if self.key_type:
                    assert isinstance(
                        key, (self.key_type, str)
                    ), f"key of register must be {self.key_type}, not {type(key)} "

                key = self._decorete_key(key, domain)
                if len(info):
                    self.D[key] = (value, info)
                else:
                    self.D[key] = value
            return value

        return insert

    def _decorete_key(self, key, domain):
        if domain is not None:
            assert isinstance(
                domain, str
            ), f"domian must be a str but get {type(domain)}"
            key = (domain, key)
        return key

    def get(self, key, domain=None):
        key = self._decorete_key(key, domain)
        if key not in self.D.keys():
            raise Exception(f"key: {key} not exists")
        return self.D[key]

    def has(self, key, domain=None):
        key = self._decorete_key(key, domain)
        return key in self.D.keys()

    def build(self, key, domain=None, **kwargs):
        key = self._decorete_key(key, domain)
        cls = self.D[key]
        if isinstance(cls, tuple) and callable(cls[0]):
            cls, info = cls
        return cls(**kwargs)

    def return_when_exist(self, key, domain=None):
        key = self._decorete_key(key, domain)
        if key in self.D.keys():
            return self.D[key]
        else:
            return None

ObserverRegister = BaseRegister(key_type=str, add_str_name=True)


# deprecated
def build_observer_for_quantizer(
    cls_type, bitwidth=8, granularity="tensor", symmetric=True, **kwargs
):
    cls = ObserverRegister.get(cls_type)
    return cls(bit_num=bitwidth, granularity=granularity, symmetric=symmetric, **kwargs)


def build_observer(cfg: Union[str, dict] = dict(), **kwargs) -> ObserverABC:
    """Universal Observer Builder, it can build a observer by str or dict

    Args:
        cfg (Union[str, dict], optional): cfg of observer ,it can be a str or dict. Defaults to dict().

    Returns:
        ObserverABC
    """
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    assert isinstance(cfg, dict), "cfg must be dict"
    cfg.update(**kwargs)
    if "type" in cfg.keys():
        cls_type = cfg.pop("type","minmax")
    else:
        cls_type = cfg.pop("calib_method", "minmax")
    
    if isinstance(cls_type, dict):
        cls_type = cls_type['type']
        
    cls = ObserverRegister.get(cls_type)
    return cls(**filter_args_of_cfg(cfg, cls))
build_speed_observer = build_observer