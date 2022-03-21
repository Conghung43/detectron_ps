import configparser
import cv2

def get_config_data():
    config = configparser.ConfigParser()
    config.read("config/config.cfg", encoding="utf-8")
    # config = config[section]
    return config

def write_config_data(config):
    with open('config/config.cfg', 'w') as configfile:    # save
        config.write(configfile)

def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))