import matplotlib.pyplot as plt
import random, shutil, pickle, sys, os, glob, argparse, sqlite3, secrets, time, csv, codecs, json, datetime, math, ray, gym, keras
import numpy as np
from collections import deque
from keras import Sequential
from IPython.display import display, clear_output
from PIL import Image
import pandas as pd
from keras import Model
import tensorflow as tf
from keras.layers import Layer, Dense, Conv2D, Flatten, RepeatVector,Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from keras.layers import Activation, LSTM, Bidirectional , Dropout
from keras import layers
from tabulate import tabulate
from tensorflow.keras.optimizers import Adam
from ray import air, tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ncps.tf import CfC
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override





class ChartDecorator:
    def __init__(self):
        self.balance_limit = 300
        self.standard_balance = 1000
        self.prev_trade = None
        
    def add_top_bottom_bar(self, img, draw_context):
        # Define the coor dinates of the black bar
        x1, y1 = 0, 204
        x2, y2 = 200, 224
        # Draw the black bar bottom bar
        draw_context.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        # Draw the black bar top bar
        x1, y1 = 0, 0
        x2, y2 = 224, 14
        draw_context.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))


    def draw_account_balance(self, img, draw_context, account_balance = 1000):
        balance = self.balance_limit + (account_balance - self.standard_balance)
        if balance == self.balance_limit:
            x1, y1 = 3, 204
            x2, y2 = 75 , 224
            # Draw the green bar for account
            draw_context.rectangle([x1, y1, x2, y2], fill=(110, 235, 131))

        elif balance < self.balance_limit:
            health_bar = int((balance * 75) / self.balance_limit )
            lost_bar = 75 - health_bar
            #issue 1
            # print(f"issue 1 balance {balance}")
            # print(f"health bar {health_bar}")
            
            # print(f"lost bar {lost_bar}")
            #current_balnace
            x1, y1 = 3, 204
            x2, y2 = max(3,health_bar), 224
            draw_context.rectangle([x1, y1, x2, y2], fill=(110, 235, 131))
            #lost bar
            x1, y1 = health_bar, 204
            x2, y2 = 75 , 224
            draw_context.rectangle([x1, y1, x2, y2], fill=(255, 0, 0))
        elif balance > self.balance_limit:
            x1, y1 = 3, 204
            x2, y2 = 75 , 224
            # Draw the green bar for account
            draw_context.rectangle([x1, y1, x2, y2], fill=(110, 235, 131)) 
            #Draw the profit section on account_bar
            profit = int(((balance - self.balance_limit) * 75) / self.balance_limit)

            x1, y1 = 80, 204
            x2, y2 = 80 + profit , 224
            # Draw the green bar for account
            draw_context.rectangle([x1, y1, x2, y2], fill=(0,0,255))# fill=(71, 44, 27)) 

        #draw start separator
        #draw Account Separator and end separator
        x1, y1 = 0, 204
        x2, y2 = 3 , 224
        # Draw the green bar for account
        draw_context.rectangle([x1, y1, x2, y2], fill=(255, 87, 20))     

        #draw Account Separator and end separator
        x1, y1 = 76, 204
        x2, y2 = 80 , 224
        # Draw the green bar for account
        draw_context.rectangle([x1, y1, x2, y2], fill=(255, 87, 20)) 

        #draw Account Separator and end separator
        x1, y1 = 155, 204
        x2, y2 = 160 , 224
        # Draw the green bar for account
        draw_context.rectangle([x1, y1, x2, y2], fill=(255, 87, 20)) 

    def draw_current_trade(self,img, draw_context, position=None):
        if position == "buy":
            x1, y1 = 160, 204
            x2, y2 = 175, 224
            draw_context.rectangle([x1, y1, x2, y2], fill=(255, 255, 0)) 
        elif position == "sell":
            x1, y1 = 180, 204
            x2, y2 = 195, 224
            draw_context.rectangle([x1, y1, x2, y2], fill=(0, 255, 0))  
        else:
            x1, y1 = 200, 204
            x2, y2 = 215, 224
            draw_context.rectangle([x1, y1, x2, y2], fill=(255,0,255)) 

        #Draw position separator icons
        x1, y1 = 175, 204
        x2, y2 = 180, 224
        draw_context.rectangle([x1, y1, x2, y2], fill=(255, 255, 255)) 
        x1, y1 = 195, 204
        x2, y2 = 200, 224
        draw_context.rectangle([x1, y1, x2, y2], fill=(255, 255, 255)) 
        x1, y1 = 215, 204
        x2, y2 = 220, 224
        draw_context.rectangle([x1, y1, x2, y2], fill=(255, 255, 255)) 

    def draw_profit_bar(self, img, draw_context, profit = 0, account_balance = 1000, position = None):
        pnl_parts = int(abs(profit) / 10)
        for segment in range(0, 200, 10):
            f = segment + 10
            x1, y1 = segment+10, 0
            x2, y2 = f + 1, 9
            draw_context.rectangle([x1, y1, x2, y2], fill=(255, 255, 255)) 
            #display profit on the chart
            if profit > 0 and pnl_parts > 0 and segment >= 100:
                if segment < pnl_parts * 10 + 100:
                    x1, y1 = segment, 0
                    x2, y2 = segment + 10, 9
                    draw_context.rectangle([x1, y1, x2, y2], fill=(27, 152, 224)) 
            elif profit < 0 and pnl_parts > 0 and segment <= 90 and segment >= 100 - abs(profit):           
                x1, y1 = segment, 0
                x2, y2 = segment + 10, 9
                draw_context.rectangle([x1, y1, x2, y2], fill=(255, 0, 0))  
            #draw progressive account balance below profit bar
            account_balance_bar = (200*account_balance)/2000
            x1, y1 = 0, 10
            x2, y2 = account_balance_bar, 14
            draw_context.rectangle([x1, y1, x2, y2], fill=(255, 119, 0)) 
            #Add a closing indication at the top of the 
            if position == "close" and profit > 0 and self.prev_trade[2] == 1:
                x1, y1 = 200, 0
                x2, y2 = 224, 14
                draw_context.rectangle([x1, y1, x2, y2], fill=(255, 255, 0))
            elif position == "close" and profit > 0 and self.prev_trade == 2:
                x1, y1 = 200, 0
                x2, y2 = 224, 14
                draw_context.rectangle([x1, y1, x2, y2], fill=(0, 255, 0))
            elif position == "close" and profit < 0 and self.prev_trade in [2,1]:
                x1, y1 = 200, 0
                x2, y2 = 224, 14
                draw_context.rectangle([x1, y1, x2, y2], fill=(255,0, 0))
       
    def draw_buy_bar(self, img2, start_lines, end_lines, position = None, profit = 0, prev_trade=None, account_balance = 1100,highest=0 ):
        self.prev_trade = prev_trade
        
        from PIL import Image, ImageDraw
        d = ImageDraw.Draw(img2)
        
        self.add_top_bottom_bar(img2, d)
        self.draw_account_balance(img2, d, account_balance)

        if position == "sell" and 2 in [self.prev_trade[3]]:
            #draw icon for trade setup
            self.draw_current_trade(img2, d, "sell")
            fixed_ratio = False
            #issue fix bug 
            #a situation where there is profit but since the trade was entered there were pertuabations of zooming and changing of window prices indicated on the subwindow
            #fix this by giving a ratio use profit to check if we are profitable, then check if end lines are lower than start lines , use a ratio of 1px for 3 points
            #when scaling the bar check the distance remaining behind you at the top of the window. start from index 14 your calculations
            if profit >= 0 and start_lines["ask_line"] > end_lines["ask_line"] :
                # print(f"\nRatio bug found {end_lines}\n")
                ratio = 15 #int((start_lines["ask_line"] - end_lines["ask_line"] ) / 3)
                if end_lines["ask_line"] - ratio < 14:
                    start_lines["ask_line"] = 14
                    start_lines["bid_line"] = 18
                else:
                    start_lines["ask_line"] = end_lines["ask_line"] - ratio
                    start_lines["bid_line"] = end_lines["bid_line"] - ratio
                fixed_ratio = True

            # Draw indicator of trade start
            
            line_color = (200,200,200)
            top = (200, end_lines["ask_line"])
            bottom = (200, end_lines["bid_line"])
            d.line([top, bottom], fill=line_color, width=20)

            #Draw start marker for the trade
            line_color = (0, 255, 0)
            top = (200, start_lines["ask_line"])
            bottom = (200, start_lines["bid_line"])
            d.line([top, bottom], fill=line_color, width=30) 
            
            #Draw start line for trade entry
            line_color = (0, 255, 0)#(252, 81, 48)
            top = (50, start_lines["ask_line"])
            bottom = (50, start_lines["bid_line"])
            d.line([top, bottom], fill=line_color, width=250) 

            #Draw profit indicator of sell
            if profit < 0:
                #Draw the main bar of the trade position
                if fixed_ratio:
                    line_color = (255, int(ratio/2+10), 0)
                    top = (200, start_lines["ask_line"])
                    bottom = (200,end_lines["bid_line"])
                    d.line([top, bottom], fill=line_color, width=int(ration/5)+10)
                    fixed_ratio = False
                else:                    
                    line_color = (255, 0, 0)
                    top = (200, start_lines["ask_line"])
                    bottom = (200,end_lines["bid_line"])
                    d.line([top, bottom], fill=line_color, width=10)
                
            else:
              #fix the highest issue to be visible on the map
                if profit > 30:
                    line_color = (255, 0 , 0)
                    top = (200, start_lines["ask_line"])
                    profit_line = abs(start_lines["ask_line"] - end_lines["bid_line"])
                    profit_line = int((profit_line * highest)/profit)
                    bottom = (200, start_lines["bid_line"] + profit_line)
                    d.line([top, bottom], fill=line_color, width=5)

                
                line_color = (0, 255, 0)
                top = (200, start_lines["ask_line"])
                bottom = (200,end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10)
            #Draw loss indicator for sell
            self.draw_profit_bar(img2, d, profit)
        elif position == "buy" and 1 in [self.prev_trade[3]]:
            #issue fix bug 2
            if profit >= 0 and end_lines["ask_line"] > start_lines["ask_line"]:
                ratio = 15
                if end_lines["bid_line"] + ratio > 204:
                    start_lines["ask_line"] = 200
                    start_lines["bid_line"] = 204
                else:
                    start_lines["ask_line"] = end_lines["ask_line"] + ratio
                    start_lines["bid_line"] = end_lines["bid_line"] + ratio         
            
            #draw bottom right trade status sell , buy , close
            self.draw_current_trade(img2, d, "buy") # 
            #Draw indicator of trade start
            #Draw end marker
            line_color = (65, 64, 102)
            top = (200, end_lines["ask_line"])
            bottom = (200, end_lines["bid_line"])
            d.line([top, bottom], fill=line_color, width=20) 

            #Draw start marker for buy
            line_color = (255, 255, 0)
            top = (200, start_lines["ask_line"])
            bottom = (200, start_lines["bid_line"])
            d.line([top, bottom], fill=line_color, width=30) 

            #Draw start buy line across the chart
            line_color = (255, 255, 0)#(255,0,200)
            top = (50, start_lines["ask_line"])
            bottom = (50, start_lines["bid_line"])
            d.line([top, bottom], fill=line_color, width=250)         


            if profit < 0:
                line_color = (255, 0, 0)
                top = (200, start_lines["ask_line"])
                bottom = (200, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10)
            else:
                #fix highest price on chart
                if profit > 30:
                    line_color = (255, 0 , 0)
                    top = (200, start_lines["ask_line"])
                    profit_line = abs(end_lines["bid_line"] - start_lines["ask_line"])

                    profit_line = int((profit_line * highest)/profit)
                    bottom = (200, start_lines["ask_line"] - profit_line)
                    d.line([top, bottom], fill=line_color, width=5) 
                
                line_color = (255, 255, 0)
                top = (200, start_lines["ask_line"])
                bottom = (200, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10)

            self.draw_profit_bar(img2, d, profit)


        elif position in ["buy", "sell"] and self.prev_trade[3] == 0:
            # print(f"Draw buy bar \n\nElse{position}\n\n:")
            #draw main vertical bar
            self.draw_current_trade(img2, d, "close")

            line_color = (255,0,255)
            top = (200, start_lines["ask_line"])
            bottom = (200, end_lines["bid_line"])
            d.line([top, bottom], fill=line_color, width=15)
            #draw end bar
            line_color = (255,0,255)
            top = (200, end_lines["ask_line"])
            bottom = (200, end_lines["bid_line"])
            d.line([top, bottom], fill=line_color, width=30) 
            #Draw start marker and line
            line_color = (255,0,255)
            top = (200, start_lines["ask_line"])
            bottom = (200, start_lines["bid_line"])
            d.line([top, bottom], fill=line_color, width=30)

            line_color = (2, 8, 135)
            top = (50, start_lines["ask_line"])
            bottom = (50, start_lines["bid_line"])
            d.line([top, bottom], fill=line_color, width=250)

            # self.draw_profit_bar(img2, d, profit)

            if self.prev_trade[2] == 2 and profit > 0:   
                #print right bar for the trade that has closed
                line_color = (0, 255, 0)
                top = (213, start_lines["ask_line"])
                bottom = (213, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10) 

                line_color = (241, 254, 198)
                top = (190, start_lines["ask_line"])
                bottom = (190, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10)             

            elif self.prev_trade[2] == 2 and profit < 0:
                line_color = (0, 255, 0)
                top = (213, start_lines["ask_line"])
                bottom = (213, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10) 

                line_color = (255,0,0)
                top = (190, start_lines["ask_line"])
                bottom = (190, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10)    
            elif self.prev_trade[2] == 1 and profit > 0:
                # print(f"\n\nHey I was here\n\n{self.prev_trade}")
                
                line_color = (255, 255, 0)
                top = (213, start_lines["ask_line"])
                bottom = (213, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10) 

                line_color = (241, 254, 198)
                top = (190, start_lines["ask_line"])
                bottom = (190, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10)    

            elif self.prev_trade[2] == 1 and profit < 0:
                line_color = (255, 255, 0)
                top = (213, start_lines["ask_line"])
                bottom = (213, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10) 

                line_color = (255,0,0)
                top = (190, start_lines["ask_line"])
                bottom = (190, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10) 
        else :#:position == "close" and self.prev_trade[2] == 0:
            self.draw_current_trade(img2, d, "close")
            # print("I run for position Close")
            line_color = (255,0,255)
            top = (200, end_lines["ask_line"])
            bottom = (200, end_lines["bid_line"])
            d.line([top, bottom], fill=line_color, width=30)  

        self.draw_profit_bar(img2, d, profit, account_balance, position)
        return img2



class Spec:
    def __init__(self, id = None):
        self.id = int(list(np.random.random(1))[0]*1000000)
        self.max_episode_steps = 200
        
class ForexCustomEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self.m1short = config["m1short"]
        self.m5short = config["m5short"]
        self.m1long = config["m1long"]
        self.m5long = config["m5long"]
        self.chart_decorator = config["chart_decorator"]

        self.trade_queue_m5_short = deque(maxlen=4)
        self.image_queue_m5_short = deque(maxlen=4)
        self.image_queue_m5_long = deque(maxlen=4)
        self.trade_queue_m5_long = deque(maxlen=4)
        self.dataset_directory = "M1M5Charts/"#"colabM1M5/episode2/"
        self.account_balance = 1000
        self.env_draw = config["env_draw"]
        self.chart_decorator = chart_decorator
        
        self.prev_trade = deque(maxlen=4)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        
        self.current_step = 4
        self.action = 0
        self.position = "close"


        self.spec = Spec()

        self.reward_ma = deque(maxlen=3)
        self.reward_Q = deque(maxlen=2)
        self.observation_space = gym.spaces.Box(low=0, high=255,shape=(224,224,3))
        self.action_space = gym.spaces.Discrete(3)



    def get_ask_bid_lines(self, image_path):
        ask_bid_lines = {"ask_line":None, "bid_line":None, "key": 0}
        img = Image.open(f"{self.dataset_directory}{image_path}")
        img_data = np.array(img)
        last_part = Image.fromarray(img_data[:,200:201,:])
        img1_pixels = last_part.load()
        ask_color = (255,0,0)
        bid_color = (119, 136, 153)
        for y in range(224):
            if img1_pixels[0,y] == ask_color:
                ask_bid_lines["ask_line"] = y
            if img1_pixels[0,y] == bid_color:
                ask_bid_lines["bid_line"] = y
        if ask_bid_lines["ask_line"] == None:
            # print("Gotcha")
            ask_bid_lines["ask_line"] = ask_bid_lines["bid_line"] + 3
            ask_bid_lines["bid_line"] = ask_bid_lines["bid_line"] + 6
            
        return ask_bid_lines , img        
    
    def reset(self, current_step = 1, step = 50, seed = 0):
        self.current_step = current_step
        self.step_size = step
        # print(f"DBG reset {self.m1short.iloc[self.current_step - 4]['image_path']}")

        linesq_m5_short, imageq_m5_short = self.get_ask_bid_lines(f"{self.m5short.iloc[self.current_step - 4]['image_path']}")
        linesq_m5_short["key"] = current_step - 4
        linest_m5_short, imaget_m5_short = self.get_ask_bid_lines(f"{self.m5short.iloc[self.current_step - 3]['image_path']}")
        linest_m5_short["key"] = current_step - 3
        liness_m5_short, images_m5_short = self.get_ask_bid_lines(f"{self.m5short.iloc[self.current_step - 2]['image_path']}")
        liness_m5_short["key"] = current_step - 2
        linesc_m5_short, imagec_m5_short = self.get_ask_bid_lines(f"{self.m5short.iloc[self.current_step - 1]['image_path']}")
        linesc_m5_short["key"] = current_step - 1
        
        linesq_m5_long, imageq_m5_long = self.get_ask_bid_lines(f"{self.m5long.iloc[self.current_step - 4]['image_path']}")
        linesq_m5_long["key"] = current_step - 4
        linest_m5_long, imaget_m5_long = self.get_ask_bid_lines(f"{self.m5long.iloc[self.current_step - 3]['image_path']}")
        linest_m5_long["key"] = current_step - 3
        liness_m5_long, images_m5_long = self.get_ask_bid_lines(f"{self.m5long.iloc[self.current_step - 2]['image_path']}")
        liness_m5_long["key"] = current_step - 2
        linesc_m5_long, imagec_m5_long = self.get_ask_bid_lines(f"{self.m5long.iloc[self.current_step - 1]['image_path']}")
        linesc_m5_long["key"] = current_step - 1

        self.trade_queue_m5_short.append(linesq_m5_short)
        self.trade_queue_m5_short.append(linest_m5_short)
        self.trade_queue_m5_short.append(liness_m5_short)
        self.trade_queue_m5_short.append(linesc_m5_short)
        self.image_queue_m5_short.append(imageq_m5_short)
        self.image_queue_m5_short.append(imaget_m5_short)
        self.image_queue_m5_short.append(images_m5_short)
        self.image_queue_m5_short.append(imagec_m5_short)         
        
        self.trade_queue_m5_long.append(linesq_m5_long)
        self.trade_queue_m5_long.append(linest_m5_long)
        self.trade_queue_m5_long.append(liness_m5_long)
        self.trade_queue_m5_long.append(linesc_m5_long)
        self.image_queue_m5_long.append(imageq_m5_long)
        self.image_queue_m5_long.append(imaget_m5_long)
        self.image_queue_m5_long.append(images_m5_long)
        self.image_queue_m5_long.append(imagec_m5_long)        
        
        self.done = 0
        self.reward = 0
        self.position = "close"
        self.reset_current_trade()
        
        next_state = self.get_obs()
        self.episode_profit = 0
        self.prev_trade = deque(maxlen=4)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        self.action = 0
        self.reward_ma = deque(maxlen=3)
        self.reward_Q = deque(maxlen=3)

        #reset standard limit in chart decorator
        self.chart_decorator.standard_limit = self.account_balance
        
        return next_state#self.current_trade 
        
    def reset_current_trade(self ):
        self.current_trade = {}
        self.current_trade["current_step"] = self.current_step
        self.current_trade["ask_bid_lines_m5_short"] = None
        self.current_trade["ask_bid_lines_m5_long"] = None
        self.current_trade["entry_price"] = None
        self.current_trade["current_price"] = None
        self.current_trade["timesteps"] = 0
        self.current_trade["position"] = "close"
        self.current_trade["profit"] = 0
        self.current_trade["parent_ask_bid_lines_m5_short"] = None
        self.current_trade["parent_ask_bid_lines_m5_long"] = None
        self.current_trade["reward"] = 0
        self.current_trade["balance"] = self.account_balance
        self.current_trade["episode_profit"] = 0
        #this code is to add dimensions to 8 of them 4 for m1short 4 for m5short remaining to draw on m5 charts
        self.current_trade["ask_bid_lines_m5_short"] = None
        
        self.current_trade["highest"] = 0
        self.current_trade["price_ma"] = deque(maxlen=3)
        self.current_trade["reward_ma"] = deque(maxlen=3)
        self.account_balance = 1000
        self.old_balance = 1000
        
        
    def display_stacked_horizontally(self,imageq, imaget,images,imagec):
        # # Calculate the required dimensions for the new image
        new_width = imageq.width * 4
        new_height = imageq.height
        # # Create a new image with the required dimensions
        new_image = Image.new("RGB", (new_width, new_height))
        # # Paste the individual images side by side
        new_image.paste(imageq, (0, 0))
        new_image.paste(imaget, (imageq.width, 0))
        new_image.paste(images, (imageq.width + imaget.width, 0))
        new_image.paste(imagec, (imageq.width + imaget.width + images.width, 0))
        

    
    def reset_collection(self, t_q, i_q):
        _queue = deque(maxlen=4)
        for i in range(0, 4):
            _lines_c, _image_c = self.get_ask_bid_lines(f"{self.dataset_directory}{self.m1short.iloc[t_q[i]['key']]['image_path']}")
            _image_c = self.draw_buy_bar(_image_c, _lines_c, _lines_c, position="close", profit = 0)
            _queue.append(_image_c)
        return t_q, _queue
    
    def draw_collection(self,q, q2):
        # # Calculate the required dimensions for the new image
        new_width = q[0].width * 4
        new_height =q[0].height
        # # Create a new image with the required dimensions
        new_image = Image.new("RGB", (new_width, new_height))
        # # Paste the individual images side by side
        # new_image.paste(q[0], (0, 0))
        # new_image.paste(q2[1], (0, 0))
        # new_image.paste(q2[2], (q[0].width, 0))
        new_image.paste(q2[3], (q[0].width + q[0].width, 0))
        # new_image.paste(q[3], (q[0].width + q[0].width + q[0].width, 0))
        display(q2[3])
        
    def draw_buy_bar(self, current_image, start_lines, end_lines, position = None, profit = 0, prev_trade=None, highest=0):        
        return self.chart_decorator.draw_buy_bar(current_image, start_lines, end_lines, self.position, profit, prev_trade, self.account_balance, highest)

    def enter_trade(self, ask_bid_lines_m1_short=None, ask_bid_lines_m5_short = None, ask_bid_lines_m1_long = None, ask_bid_lines_m5_long = None, current_image_m1_short=None, current_image_m5_short=None, current_image_m1_long=None, current_image_m5_long = None):
        self.current_trade["position"] = self.position
        self.current_trade["ask_bid_lines_m5_short"] = ask_bid_lines_m5_short
        self.current_trade["ask_bid_lines_m5_long"] = ask_bid_lines_m5_long
        self.current_trade["entry_price"] = float(f"{self.m5short.iloc[self.current_step]['Ask']}") if self.position == "buy" else float(f"{self.m5short.iloc[self.current_step]['Bid']}")
        self.current_trade["profit"] = 0
        self.current_trade["current_price"] = float(f"{self.m5short.iloc[self.current_step]['Ask']}")
        self.current_trade["timesteps"] = 1
        self.current_trade["parent_ask_bid_lines_m5_short"] = ask_bid_lines_m5_short
        self.current_trade["parent_ask_bid_lines_m5_long"] = ask_bid_lines_m5_long
        self.current_trade["balance"] = self.account_balance
        self.current_trade["highest"] = 0
        self.current_trade["price_ma"] = deque(maxlen=2)
        self.current_trade["reward_ma"] = deque(maxlen=2)
        self.current_trade["price_ma"].append(0)
        self.current_trade["price_ma"].append(0)
        self.current_trade["price_ma"].append(0)
        self.current_trade["reward_ma"].append(0.10)
        self.current_trade["reward_ma"].append(0.10)
        self.current_trade["reward_ma"].append(0.10)
        current_image_m5_short = self.draw_buy_bar(current_image_m5_short, ask_bid_lines_m5_short, ask_bid_lines_m5_short, self.position , self.current_trade["profit"], self.prev_trade, self.current_trade["highest"] )
        current_image_m5_long = self.draw_buy_bar(current_image_m5_long, ask_bid_lines_m5_long, ask_bid_lines_m5_long, self.position, self.current_trade["profit"], self.prev_trade, self.current_trade["highest"])
        self.trade_queue_m5_short.append(ask_bid_lines_m5_short)
        self.trade_queue_m5_long.append(ask_bid_lines_m5_long)
        self.image_queue_m5_short.append(current_image_m5_short)  
        self.image_queue_m5_long.append(current_image_m5_long)
        if self.env_draw:
            
            self.draw_collection(self.image_queue_m5_short,self.image_queue_m5_long) 
            # self.draw_collection(self.image_queue_m5_short,self.image_queue_m5_long) 
        self.reward = 0.1# self.get_reward(self.current_trade) #self.account_balance/self.account_balance
        self.reward_Q.append(self.reward)
        self.reward_Q.append(self.reward)
        self.reward_Q.append(self.reward)
        
        self.current_trade["reward"]  = self.reward
        # print(f"DBG: enter trade_ current_trade {self.current_trade}")
        
    def enter_idle_mode(self, ask_bid_lines_m1_short=None, ask_bid_lines_m5_short = None, ask_bid_lines_m1_long = None, ask_bid_lines_m5_long = None, current_image_m1_short=None, current_image_m5_short=None, current_image_m1_long=None, current_image_m5_long = None):
        
        self.current_trade["ask_bid_lines_m5_short"] = ask_bid_lines_m5_short
        self.current_trade["ask_bid_lines_m5_long"] = ask_bid_lines_m5_long
        self.current_trade["entry_price"] = float(f"{self.m5short.iloc[self.current_step]['Ask']}")
        self.current_trade["current_price"] = float(f"{self.m5short.iloc[self.current_step]['Ask']}")
        self.current_trade["position"] = self.position
        self.current_trade["reward"] = 0
        self.current_trade["timesteps"] = 1
        self.current_trade["profit"] = -3
        self.current_trade["parent_ask_bid_lines_m5_short"] = None
        self.current_trade["parent_ask_bid_lines_m5_long"] = None
        self.current_trade["balance"] = self.account_balance
        
        self.current_trade["price_ma"].append(0.10)
        self.current_trade["price_ma"].append(0.10)
        self.current_trade["price_ma"].append(0.10)
        self.current_trade["price_ma"].append(0.10)
        
        self.current_trade["reward_ma"].append(0.00)
        self.current_trade["reward_ma"].append(0.00)
        self.current_trade["reward_ma"].append(0.00)
        self.current_trade["reward_ma"].append(0.00)
        
        current_image_m5_short = self.draw_buy_bar(current_image_m5_short, ask_bid_lines_m5_short, ask_bid_lines_m5_short, self.position, self.current_trade["profit"], self.prev_trade, self.current_trade["highest"])
        current_image_m5_long = self.draw_buy_bar(current_image_m5_long, ask_bid_lines_m5_long, ask_bid_lines_m5_long, self.position, self.current_trade["profit"], self.prev_trade, self.current_trade["highest"])
        self.trade_queue_m5_short.append(ask_bid_lines_m5_short)
        self.trade_queue_m5_long.append(ask_bid_lines_m5_long)
        self.image_queue_m5_short.append(current_image_m5_short)  
        self.image_queue_m5_long.append(current_image_m5_long)
        if self.env_draw:
            self.draw_collection(self.image_queue_m5_short,self.image_queue_m5_long) 
            # self.draw_collection(self.image_queue_m5_short,self.image_queue_m5_long) 
        self.reward = 0#self.get_reward(self.current_trade)
        self.current_trade["reward"] = self.reward
        # print(f"DBG: enter idle_ current_trade {self.current_trade}")
    def hold_position(self, ask_bid_lines_m1_short=None, ask_bid_lines_m5_short = None, ask_bid_lines_m1_long = None, ask_bid_lines_m5_long = None, current_image_m1_short=None, current_image_m5_short=None, current_image_m1_long=None, current_image_m5_long = None):
        
        self.current_trade["current_price"] = float(f"{self.m5short.iloc[self.current_step]['Bid']}") if self.position == "buy" else float(f"{self.m5short.iloc[self.current_step]['Ask']}")
        self.current_trade["profit"] = self.current_trade["current_price"] - self.current_trade["entry_price"] if self.position == "buy" else self.current_trade["entry_price"] - self.current_trade["current_price"]
        self.current_trade["timesteps"] += 1
        self.current_trade["current_timestep"] = self.current_step
        self.current_trade["balance"] = self.account_balance
        self.current_trade["highest"] = self.current_trade["profit"] if self.current_trade["profit"] > self.current_trade["highest"] else self.current_trade["highest"]
        #self.current_trade["highest"] -= 15

        current_image_m5_short = self.draw_buy_bar(current_image_m5_short, self.current_trade["parent_ask_bid_lines_m5_short"], ask_bid_lines_m5_short, self.position , self.current_trade["profit"], self.prev_trade, self.current_trade["highest"])
        current_image_m5_long = self.draw_buy_bar(current_image_m5_long, self.current_trade["parent_ask_bid_lines_m5_long"], ask_bid_lines_m5_long, self.position, self.current_trade["profit"], self.prev_trade, self.current_trade["highest"])
        self.trade_queue_m5_short.append(ask_bid_lines_m5_short)
        self.trade_queue_m5_long.append(ask_bid_lines_m5_long)
        self.image_queue_m5_short.append(current_image_m5_short)
        self.image_queue_m5_long.append(current_image_m5_long)
        #fix price variance with ma for each trade
        self.current_trade["price_ma"].append(self.current_trade["profit"])
        
        if len(self.current_trade["price_ma"]) < 2:
            if self.current_trade["profit"] < 0 and self.current_trade > -10:
                self.current_trade["price_ma"].append(1)
                self.current_trade["price_ma"].append(1)
                # self.current_trade["price_ma"].append(1)
                # self.current_trade["price_ma"].append(1)
                
            elif self.current_trade["price_ma"] > 0 and self.current_trade["profit"] > 0:
                self.current_trade["price_ma"].append(self.current_trade["profit"])
                self.current_trade["price_ma"].append(self.current_trade["profit"])
                # self.current_trade["price_ma"].append(self.current_trade["profit"])
                # self.current_trade["price_ma"].append(self.current_trade["profit"])
                # print(f"DBG: hold_position {self.current_trade['price_ma']}")
        
        if self.env_draw:
            self.draw_collection(self.image_queue_m5_short,self.image_queue_m5_long) 
            # self.draw_collection(self.image_queue_m5_short,self.image_queue_m5_long) 
        if self.current_trade["profit"] >= 0:
            self.reward = 0.2#self.get_reward(self.current_trade)#(self.account_balance + self.current_trade["profit"])/self.account_balance
            self.current_trade["reward"] = self.reward
        else:#refactor introduce a penalty for the magnitude of losses
            if self.current_trade["profit"] > -50:
                self.done = True
                self.reward = -1
            else:
                self.reward = -1 * self.get_reward(abs(self.current_trade["profit"]))


        if self.current_trade["profit"] * 80 < self.current_trade["highest"]:
            self.reward = - (self.current_trade["highest"] - self.current_trade["profit"])/(1+self.current_trade["highest"])
            
        # print(f"DBG::hold positio   n_ current_trade {self.current_trade}")
        self.current_trade["reward_ma"].append(self.reward)
        # self.reward = 0#sum(self.current_trade["reward_ma"])/len(self.current_trade["reward_ma"])
        self.reward_Q.append(self.reward)
        # self.reward = sum(self.reward_Q)/len(self.reward_Q)
        self.current_trade["reward"] = self.reward

    def get_reward(self, profit):
        if profit < 50:
            return profit / 50
        elif profit < 100:
            return profit / 100
        elif profit < 150:
            return profit / 150
        elif profit < 200:
            return profit / 200
        elif profit < 250:
            return profit / 250
        elif profit < 300:
            return profit / 300
        elif profit < 350 :
            return profit / 350
        elif profit < 400:
            return profit / 400
        elif profit < 450:
            return profit / 450
        elif profit < 500:
            return profit / 500
    
    def close_position(self, ask_bid_lines_m1_short=None, ask_bid_lines_m5_short = None, ask_bid_lines_m1_long = None, ask_bid_lines_m5_long = None, current_image_m1_short=None, current_image_m5_short=None, current_image_m1_long=None, current_image_m5_long = None, prev_trade = None):
        # print(f"self.position {self.position}")
        self.current_trade["current_price"] = float(f"{self.m5short.iloc[self.current_step]['Bid']}") if self.position == "buy" else float(f"{self.m5short.iloc[self.current_step]['Ask']}")
        self.current_trade["profit"] = self.current_trade["current_price"] - self.current_trade["entry_price"] if self.position == "buy" else self.current_trade["entry_price"] - self.current_trade["current_price"]
        self.current_trade["highest"] = self.current_trade["profit"] if self.current_trade["profit"] > self.current_trade["highest"] else self.current_trade["highest"]
        #self.current_trade["highest"] += 5

        current_image_m5_short = self.draw_buy_bar(current_image_m5_short, self.parent_ask_bid_lines_m5_short, ask_bid_lines_m5_short, self.position, self.current_trade["profit"] , prev_trade, self.current_trade["highest"])
        current_image_m5_long = self.draw_buy_bar(current_image_m5_long, self.parent_ask_bid_lines_m5_long, ask_bid_lines_m5_long, self.position, self.current_trade["profit"], prev_trade, self.current_trade["highest"])
        self.trade_queue_m5_short.append(ask_bid_lines_m5_short)
        self.trade_queue_m5_long.append(ask_bid_lines_m5_long)
        self.image_queue_m5_short.append(current_image_m5_short)
        self.image_queue_m5_long.append(current_image_m5_long)
        self.account_balance += self.current_trade["profit"]
        #fix high variance for the price
        self.current_trade["price_ma"].append(self.current_trade["profit"])
        
        if self.env_draw:
            self.draw_collection(self.image_queue_m5_short,self.image_queue_m5_long) 
            # self.draw_collection(self.image_queue_m5_short,self.image_queue_m5_long) 
        if self.current_trade["profit"] >= 0:
            if self.current_trade["profit"] * 80 < self.current_trade["highest"]:
                self.reward = - (self.current_trade["highest"] - self.current_trade["profit"])/(self.current_trade["highest"]+1)#self.get_reward(self.current_trade["profit"]
            else:
                self.reward = self.get_reward(self.current_trade["profit"])
        else:
            self.reward =  -self.get_reward(abs(self.current_trade["profit"]))
        
        self.current_trade["reward_ma"].append(self.reward)
        self.current_trade["reward"] = self.reward        
        
        self.current_trade["current_timestep"] = self.current_step
        self.current_trade["timesteps"] += 1
        self.current_trade["position"] = "close"
        self.current_trade["balance"] = self.account_balance
        self.current_trade["episode_profit"] = self.episode_profit
        self.position = "close"
        self.parent_ask_bid = None
        
 
    def step(self, action):
        self.reward = 0
        self.done = False

        ask_bid_lines_m5_short, current_image_m5_short = self.get_ask_bid_lines(f"{self.m5short.iloc[self.current_step]['image_path']}")
        ask_bid_lines_m5_long, current_image_m5_long = self.get_ask_bid_lines(f"{self.m5long.iloc[self.current_step]['image_path']}")
        ask_bid_lines_m5_short["key"] = self.current_step
        ask_bid_lines_m5_long["key"] = self.current_step
        
        
        
        if self.position == "close":
            if action == 1:
                self.position = "buy"
                self.prev_trade.append(1)
                self.parent_ask_bid_lines_m5_short = ask_bid_lines_m5_short
                self.parent_ask_bid_lines_m5_long = ask_bid_lines_m5_long
                
                self.enter_trade(ask_bid_lines_m5_short = ask_bid_lines_m5_short, ask_bid_lines_m5_long = ask_bid_lines_m5_long, current_image_m5_short = current_image_m5_short, current_image_m5_long = current_image_m5_long)
            elif action == 2:
                self.position = "sell"
                self.prev_trade.append(2)
                self.parent_ask_bid_lines_m5_short = ask_bid_lines_m5_short
                self.parent_ask_bid_lines_m5_long = ask_bid_lines_m5_long
                self.enter_trade(ask_bid_lines_m5_short = ask_bid_lines_m5_short, ask_bid_lines_m5_long = ask_bid_lines_m5_long, current_image_m5_short = current_image_m5_short, current_image_m5_long = current_image_m5_long)
            elif action == 0:
                self.position = "close"
                self.prev_trade.append(0)
                self.parent_ask_bid_lines_m5_short = ask_bid_lines_m5_short
                self.parent_ask_bid_lines_m5_long = ask_bid_lines_m5_long
                self.enter_idle_mode(ask_bid_lines_m5_short = ask_bid_lines_m5_short, ask_bid_lines_m5_long = ask_bid_lines_m5_long, current_image_m5_short = current_image_m5_short, current_image_m5_long = current_image_m5_long)
        elif self.position == "buy":
            if action == 1:
                self.prev_trade.append(1)
                self.hold_position(ask_bid_lines_m5_short = ask_bid_lines_m5_short, ask_bid_lines_m5_long = ask_bid_lines_m5_long, current_image_m5_short = current_image_m5_short, current_image_m5_long = current_image_m5_long)
            elif action in [2]:
                self.prev_trade.append(0)
                self.close_position(ask_bid_lines_m5_short = ask_bid_lines_m5_short, ask_bid_lines_m5_long = ask_bid_lines_m5_long, current_image_m5_short = current_image_m5_short, current_image_m5_long = current_image_m5_long, prev_trade = self.prev_trade)                
                #fix standard limit and standard balance on chart 
                if self.account_balance > int((2/3) * self.chart_decorator.balance_limit) + self.chart_decorator.standard_balance:
                    self.chart_decorator.standard_balance = self.account_balance
                    
            elif action in [0]:
                self.prev_trade.append(0)
                self.close_position(ask_bid_lines_m5_short = ask_bid_lines_m5_short, ask_bid_lines_m5_long = ask_bid_lines_m5_long, current_image_m5_short = current_image_m5_short, current_image_m5_long=current_image_m5_long, prev_trade = self.prev_trade)   
                
                #fix standard limit and standard balance on chart
                if self.account_balance > int((2/3) * self.chart_decorator.balance_limit) + self.chart_decorator.standard_balance:
                    self.chart_decorator.standard_balance = self.account_balance
        elif self.position == "sell":
            if action == 2:
                self.prev_trade.append(2)
                self.hold_position(ask_bid_lines_m5_short = ask_bid_lines_m5_short, ask_bid_lines_m5_long = ask_bid_lines_m5_long, current_image_m5_short = current_image_m5_short, current_image_m5_long = current_image_m5_long)
            #this reward is to penalize it not to change the position without closing the position
            elif action in [1]:
                self.prev_trade.append(0)
                self.close_position(ask_bid_lines_m5_short = ask_bid_lines_m5_short, ask_bid_lines_m5_long = ask_bid_lines_m5_long, current_image_m5_short = current_image_m5_short, current_image_m5_long = current_image_m5_long, prev_trade = self.prev_trade)
                #fix standard limit and standard balance on chart
                if self.account_balance > int((2/3) * self.chart_decorator.balance_limit) + self.chart_decorator.standard_balance:
                    self.chart_decorator.standard_balance = self.account_balance
            elif action == 0:
                self.prev_trade.append(0)
                self.close_position(ask_bid_lines_m5_short = ask_bid_lines_m5_short, ask_bid_lines_m5_long = ask_bid_lines_m5_long, current_image_m5_short = current_image_m5_short, current_image_m5_long = current_image_m5_long, prev_trade = self.prev_trade)

                if self.account_balance > int((2/3) * self.chart_decorator.balance_limit) + self.chart_decorator.standard_balance:
                    self.chart_decorator.standard_balance = self.account_balance

        self.current_step += self.step_size
        print(f"step {self.current_step}, action: {action}, profit  {self.current_trade['profit']} balance : {self.account_balance}")
        next_state = self.get_obs()
        if self.account_balance < 1000 or self.current_step > 9800:#:self.old_balance:
            self.done = True

        return next_state, self.reward, self.done, self.current_trade
    
    def get_obs(self):

        return np.asarray(self.image_queue_m5_long[3])

class ConvCfCModel(RecurrentNetwork):
    """Example of using the Keras functional API to define a RNN model."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        cell_size=64,
    ):
        super(ConvCfCModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.cell_size = cell_size

        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0] * obs_space.shape[1] * obs_space.shape[2]),
            name="inputs",
        )
        state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to CfC
        self.conv_block = tf.keras.models.Sequential(
            [
                tf.keras.Input(
                    (obs_space.shape[0] * obs_space.shape[1] * obs_space.shape[2])
                ),  # batch dimension is implicit
                tf.keras.layers.Lambda(
                    lambda x: tf.cast(x, tf.float32) / 255.0
                ),  # normalize input
                tf.keras.layers.Reshape(
                    (obs_space.shape[0], obs_space.shape[1], obs_space.shape[2])
                ),
                tf.keras.layers.Conv2D(
                    64, 5, padding="same", activation="relu", strides=2
                ),
                tf.keras.layers.Conv2D(
                    128, 5, padding="same", activation="relu", strides=2
                ),
                tf.keras.layers.Conv2D(
                    128, 5, padding="same", activation="relu", strides=2
                ),
                tf.keras.layers.Conv2D(
                    256, 5, padding="same", activation="relu", strides=2
                ),
                tf.keras.layers.GlobalAveragePooling2D(),
            ]
        )
        self.td_conv = tf.keras.layers.TimeDistributed(self.conv_block)

        dense1 = self.td_conv(input_layer)
        cfc_out, state_h = CfC(
            cell_size, return_sequences=True, return_state=True, name="cfc"
        )(
            inputs=dense1,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h],
        )

        # Postprocess CfC output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs, activation=tf.keras.activations.linear, name="logits"
        )(cfc_out)
        values = tf.keras.layers.Dense(1, activation=None, name="values")(cfc_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h],
            outputs=[logits, values, state_h],
        )
        self.rnn_model.summary()

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h = self.rnn_model([inputs, seq_lens] + state)
        return model_out, [h]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])

ModelCatalog.register_custom_model("cfc", ConvCfCModel)

env_config = {}
m5short_step_4 = pd.read_csv("final_complete_m5short_timestep_4.csv")
m1short_step_4 = pd.read_csv("final_complete_m1short_timestep_4.csv")
m1long_step_4 = pd.read_csv("final_complete_m1long_timestep_4.csv")
m5long_step_4 = pd.read_csv("final_complete_m5long_timestep_4.csv")
chart_decorator = ChartDecorator()
env_config["m1short"]= m1short_step_4
env_config["m5short"]= m5short_step_4
env_config["m1long"] = m1long_step_4
env_config["m5long"] = m5long_step_4
env_config["chart_decorator"] = chart_decorator
env_config["env_draw"] = False


register_env("forexcustomenv",  lambda env_config : ForexCustomEnv(env_config))

ppo_config = {
    "log_level":"INFO",
    "monitor":True,
    "env": "forexcustomenv",
    "env_config": env_config,
    "preprocessor_pref": None,
    "gamma": 0.99,
    "num_gpus": 0,
    "num_workers": 4,
    "num_envs_per_worker": 1,
    "create_env_on_driver": True,
    "lambda": 0.95,
    "kl_coeff": 0.5,
    "clip_rewards": True,
    "clip_param": 0.1,
    "vf_clip_param": 10.0,
    "entropy_coeff": 0.01,
    "horizon": 200,
    "rollout_fragment_length": 200,
    "sgd_minibatch_size": 100,
    "train_batch_size": 4000,
    "num_sgd_iter": 10,
    "lr":0.001,
    "recreate_failed_workers":True,
    "ignore_worker_failures":True,        
    "batch_mode": "truncate_episodes",
    "observation_filter": "NoFilter",
    

      

    "model": {
        "vf_share_layers": True,
        "custom_model": "cfc",
        "max_seq_len": 20,
        "custom_model_config": {
            "cell_size": 64,
        },
    },
    "framework": "tf2",
}
model_path = f"rl_ckpt_forex/forexcustomenv/policies/default_policy/policy_state.pkl"
algo = PPO(config=ppo_config)
if os.path.exists(model_path):
    algo.load_checkpoint(model_path)

def run_closed_loop(algo, config):

    env_config_t = {}
    m5short_step_4 = pd.read_csv("final_complete_m5short_timestep_4.csv")
    m1short_step_4 = pd.read_csv("final_complete_m1short_timestep_4.csv")
    m1long_step_4 = pd.read_csv("final_complete_m1long_timestep_4.csv")
    m5long_step_4 = pd.read_csv("final_complete_m5long_timestep_4.csv")
    chart_decorator = ChartDecorator()
    env_config_t["m1short"]= m1short_step_4
    env_config_t["m5short"]= m5short_step_4
    env_config_t["m1long"] = m1long_step_4
    env_config_t["m5long"] = m5long_step_4
    env_config_t["chart_decorator"] = chart_decorator
    env_config_t["env_draw"] = False
    
    
    counter = 0
    profits = 0
    print(" \n\n\nRunning closed loop\n\n\n\n")
    envt = ForexCustomEnv(env_config_t)
    # envt = wrap_deepmind(env)
    rnn_cell_size = ppo_config["model"]["custom_model_config"]["cell_size"]
    obs = envt.reset()
    state = init_state = [np.zeros(rnn_cell_size, np.float32)]
    headers = ["counter", "position","reward", "done", "profit","account_balance"]
    while True:
        rows = []
        action, state, _ = algo.compute_single_action(
            obs, state=state, explore=False, policy_id="default_policy"
        )
        obs, reward, done, current_trade = envt.step(action)
        row = [ counter, current_trade["position"], current_trade["reward"], done, current_trade["profit"], current_trade["balance"]]
        rows.append(row)
        print(tabulate(rows,headers=headers, tablefmt="grid"))
        
        if current_trade["position"] == "close":
            profits += current_trade["profit"]
        if done == "4546546554":
            obs = envt.reset()
            state = init_state
        if counter >= 9800:
            print(f"Total profits {profits}")
            break
        counter += 50

cont = "jjj"
env_name = "forexcustomenv"
render = "human"
hours = 20
os.makedirs(f"rl_ckpt_forex/{env_name}", exist_ok=True)

if render == "human":
    run_closed_loop(
        algo,
        ppo_config,
    )
else:
    print("Training STarted")
    start_time = time.time()
    last_eval = 0
    while True:
        info = algo.train()
        if time.time() - last_eval > 60 * 1:  # every 5 minutes print some stats
            print(f"Ran {(time.time()-start_time)/60/60:0.1f} hours")
            print(
                f"    sampled {info['info']['num_env_steps_sampled']/1000:0.0f}k steps"
            )
            print(f"    policy reward: {info['episode_reward_mean']:0.1f}")
            last_eval = time.time()
            ckpt = algo.save_checkpoint(f"rl_ckpt_forex/{env_name}")
            print(f"    saved checkpoint '{ckpt}'")
            print(f"Evaluation")
            run_closed_loop(algo,ppo_config)

        elapsed = (time.time() - start_time) / 60  # in minutes
        if elapsed > hours * 60:
            break
