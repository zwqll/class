import redis
import configparser
import threading


class W2VDAL:
    def __init__(self):
        self.redis_client = RedisOperator()

    def get_vectors(self, word):
        return self.redis_client.get(word)

    def update_word(self, word, vectors):
        return self.redis_client.set(word, vectors)

    def exists_word(self, word):
        result = self.redis_client.exists(word)
        return result


class RedisOperator(object):
    _instance_lock = threading.Lock()

    def __custom_init__(self):
        config_path = './config.ini'

        conf = configparser.ConfigParser()
        conf.read(config_path)

        self.db_host = conf.get("redis-db", "db_host")
        self.db_port = int(conf.get("redis-db", "db_port"))
        self.db_pass = conf.get("redis-db", "db_pass")
        #self.r = redis.Redis(host=self.db_host, port=self.db_port, password=self.db_pass, decode_responses=True)
        self.r = redis.Redis(host=self.db_host, port=self.db_port, decode_responses=True)

    def __new__(cls, *args, **kwargs):
        if not hasattr(RedisOperator, "_instance"):
            with RedisOperator._instance_lock:
                if not hasattr(RedisOperator, "_instance"):
                    RedisOperator._instance = object.__new__(cls)
                    RedisOperator._instance.__custom_init__()
        return RedisOperator._instance

    def get(self, key):
        return self.r.get(key)

    def set(self, key, value):
        return self.r.set(key, value)

    def exists(self, key):
        return self.r.exists(key)
