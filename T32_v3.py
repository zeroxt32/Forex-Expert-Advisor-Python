import matplotlib.pyplot as plt
import random, shutil, pickle, sys
import numpy as np
from collections import deque
from keras import Sequential
from keras.optimizers import RMSprop
from IPython.display import display
from PIL import Image
import pandas as pd
from keras import Model
import keras, os, glob
import tensorflow as tf
from keras.layers import Layer, Dense, Conv2D, Flatten, RepeatVector,Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from keras.layers import Activation, LSTM, Bidirectional , Dropout
from keras import layers
# import cv2
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import codecs
import csv
import secrets
import sqlite3
from IPython.display import display
from IPython.display import clear_output
import time
from tabulate import tabulate
import json
import datetime
from tensorflow.keras.optimizers import Adam
import math

class ChartDecorator:
    def __init__(self):
        self.balance_limit = 300
        self.standard_balance = 1000
        self.prev_trade = None
        
    def add_top_bottom_bar(self, img, draw_context):
        # Define the coordinates of the black bar
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
            x2, y2 = health_bar, 224
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
       
    def draw_buy_bar(self, img2, start_lines, end_lines, position = None, profit = 0, prev_trade=None, account_balance = 1100 ):
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
    
    
# def display_row_images(long_short_dict_now , long_short_dict_next):
from collections import deque
class ForexCustomEnv:
    def __init__(self, m1short, m5short,chart_decorator):
        self.m1short = m1short
        self.m5short = m5short
        self.trade_queue = deque(maxlen=4)
        self.image_queue = deque(maxlen=4)
        self.trade_queue_m5 = deque(maxlen=4)
        self.image_queue_m5 = deque(maxlen=4)
        self.dataset_directory = "colabM1M5/episode2/"
        self.account_balance = 1000
        self.env_draw = False
        self.chart_decorator = chart_decorator
        
        self.prev_trade = deque(maxlen=4)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        
        self.current_step = 4
        self.action = 0
        self.position = "close"

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
    
    def reset(self, current_step = 4):
        self.current_step = current_step
        linesq, imageq = self.get_ask_bid_lines(f"{self.m1short.iloc[self.current_step - 4]['image_path']}")
        linesq["key"] = current_step - 4
        linest, imaget = self.get_ask_bid_lines(f"{self.m1short.iloc[self.current_step - 3]['image_path']}")
        linest["key"] = current_step - 3
        liness, images = self.get_ask_bid_lines(f"{self.m1short.iloc[self.current_step - 2]['image_path']}")
        liness["key"] = current_step - 2
        linesc, imagec = self.get_ask_bid_lines(f"{self.m1short.iloc[self.current_step - 1]['image_path']}")
        linesc["key"] = current_step - 1
        
        
        linesq_m5, imageq_m5 = self.get_ask_bid_lines(f"{self.m5short.iloc[self.current_step - 4]['image_path']}")
        linesq_m5["key"] = current_step - 4
        linest_m5, imaget_m5 = self.get_ask_bid_lines(f"{self.m5short.iloc[self.current_step - 3]['image_path']}")
        linest_m5["key"] = current_step - 3
        liness_m5, images_m5 = self.get_ask_bid_lines(f"{self.m5short.iloc[self.current_step - 2]['image_path']}")
        liness_m5["key"] = current_step - 2
        linesc_m5, imagec_m5 = self.get_ask_bid_lines(f"{self.m5short.iloc[self.current_step - 1]['image_path']}")
        linesc_m5["key"] = current_step - 1
        
        
        
        
        self.trade_queue.append(linesq)
        self.trade_queue.append(linest)
        self.trade_queue.append(liness)
        self.trade_queue.append(linesc)
        self.image_queue.append(imageq)
        self.image_queue.append(imaget)
        self.image_queue.append(images)
        self.image_queue.append(imagec)
        
        self.trade_queue_m5.append(linesq_m5)
        self.trade_queue_m5.append(linest_m5)
        self.trade_queue_m5.append(liness_m5)
        self.trade_queue_m5.append(linesc_m5)
        self.image_queue_m5.append(imageq_m5)
        self.image_queue_m5.append(imaget_m5)
        self.image_queue_m5.append(images_m5)
        self.image_queue_m5.append(imagec_m5)         
        
        self.done = False
        self.reward = 0
        self.position = "close"
        self.reset_current_trade()
        self.account_balance = 1000
        next_state = self.get_obs()
        self.episode_profit = 0
        self.prev_trade = deque(maxlen=4)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        self.action = 0
        #reset standard limit in chart decorator
        self.chart_decorator.standard_limit = self.account_balance
        
        return next_state, self.reward, self.done, self.current_trade 
        
    def reset_current_trade(self):
        self.current_trade = {}
        self.current_trade["current_step"] = self.current_step
        self.current_trade["ask_bid_lines"] = None
        self.current_trade["ask_bid_lines_m5"] = None
        self.current_trade["entry_price"] = None
        self.current_trade["current_price"] = None
        self.current_trade["timesteps"] = 0
        self.current_trade["position"] = "close"
        self.current_trade["profit"] = 0
        self.current_trade["position"] = None
        self.current_trade["parent_ask_bid_lines"] = None
        self.current_trade["reward"] = 0
        self.current_trade["balance"] = self.account_balance
        self.current_trade["episode_profit"] = 0
        #this code is to add dimensions to 8 of them 4 for m1short 4 for m5short remaining to draw on m5 charts
        self.current_trade["ask_bid_lines_m5"] = None
        self.current_trade["parent_ask_bid_lines_m5"] = None
        self.current_trade["highest"] = 0
        
        
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
    
    def draw_collection(self,q):
        # # Calculate the required dimensions for the new image
        new_width = q[0].width * 4
        new_height =q[0].height
        # # Create a new image with the required dimensions
        new_image = Image.new("RGB", (new_width, new_height))
        # # Paste the individual images side by side
        new_image.paste(q[0], (0, 0))
        new_image.paste(q[1], (q[0].width, 0))
        new_image.paste(q[2], (q[0].width + q[0].width, 0))
        new_image.paste(q[3], (q[0].width + q[0].width + q[0].width, 0))
        display(new_image)
        
    def draw_buy_bar(self, current_image, start_lines, end_lines, position = None, profit = 0, prev_trade=None):
        # self.position = position
        # self.profit = profit
        return self.chart_decorator.draw_buy_bar(current_image, start_lines, end_lines, self.position, profit, prev_trade, self.account_balance)

    
    '''
    @description: Enters a trade in the forex environment
                  Enters the trade in the current time step, sets the position of the trade. sets the ask bid lines for the current frame
                  sets the entry price for the trade, and profit = 0. Then draws on the image the entry icon on the chart
                  and saves the updated current_image with modifications to the image queue with its corresponding ask and bid lines. 
                  then initializes the timesteps of the current trade to 0. sets the reward to 0.2
    @params
                  @position = None the position of the current_trade
                  @ask_bid_lines of the current_image
                  @current_image the current frame of the environment
                  
    '''
    
    def calculate_reward(self, profit_amount, risk_level, profit_param_a, profit_param_b, risk_param_c, risk_param_d, time_elapsed, discount_factor):
        profit_val = math.tanh(profit_param_a * profit_amount - profit_param_b)
        scaled_profit_val = profit_param_a * profit_val
        risk_val = math.exp(-risk_param_c * risk_level)
        time_val = math.exp(-discount_factor * time_elapsed)
        return (scaled_profit_val * risk_val * time_val)*10
    def get_reward(self, c_trade):
        
        return self.calculate_reward(c_trade["profit"],risk_level=0.9, profit_param_a=0.05, profit_param_b=0.9, risk_param_c=0.000005, risk_param_d=0.5, time_elapsed=c_trade["timesteps"], discount_factor=0.1)
        # Define the profit and loss thresholds
#         entry_price = c_trade["entry_price"]
#         exit_price = c_trade["current_price"]
#         highest = c_trade["highest"]
#         timesteps = c_trade["timesteps"]
#         profit = c_trade["profit"]
#         # profit_threshold = highest if highest > 40 else highest + 200
#         # Define the risk factor
#         volatility = 0.2
#         time_to_expire = (timesteps * 60) / 60  # Convert holding period to hours
#         risk_factor = (volatility * np.sqrt((timesteps * 60 )/ 60)) + 0.00001
#         #reward if profit > 0:
#         # #profit > 0 and profit < threshold
#         if profit == 0:
#             reward = 0.01
#             # print(f"GR: if {c_trade} reward {reward}")
#         # reversal in trade
#         elif highest * 0.80 >= profit and highest > 20:
#             # reward = -(1 /( profit/risk_factor*0.001))
#             reward = (-1/(((profit/risk_factor)*0.01))) 
#             print(f"GR: highest ran {c_trade} reward {reward}")
#         elif profit > 0 :#and #profit < profit_threshold :
#             reward = ((profit/risk_factor)*0.001) 
#             print(f"GR: Profit > 0, < threshold {c_trade} reward {reward}")

#         elif profit < 0:
#             reward = ((profit/risk_factor)*0.001) * 9
#             # reward = -(1 /( profit/risk_factor*0.001))   
#             print(f"GR: profit < 0 {c_trade} reward {reward}")
#         else:
#             reward = 0
#             print(f"GR: else {c_trade} reward {reward}")
#         # Return the reward
#         if reward > 1:
#             reward = 0.2 * 0.05
#         if reward < -1:
#             reward = -1 * 0.05
#         return reward
    
    
    def enter_trade(self, ask_bid_lines, ask_bid_lines_m5,current_image, current_image_m5):
        self.current_trade["position"] = self.position
        
        self.current_trade["ask_bid_lines"] = ask_bid_lines
        self.current_trade["ask_bid_lines_m5"] = ask_bid_lines_m5
        self.current_trade["entry_price"] = float(f"{self.m1short.iloc[self.current_step]['Ask']}") if self.position == "buy" else float(f"{self.m1short.iloc[self.current_step]['Bid']}")
        self.current_trade["profit"] = 0
        self.current_trade["current_price"] = float(f"{self.m1short.iloc[self.current_step]['Ask']}")
        self.current_trade["timesteps"] = 0
        self.current_trade["reward"] = 0.4
        self.current_trade["parent_ask_bid_lines"] = ask_bid_lines
        self.current_trade["parent_ask_bid_lines_m5"] = ask_bid_lines_m5
        self.current_trade["balance"] = self.account_balance
        self.current_trade["highest"] = 0
        #self.current_trade["highest"] += 5

        current_image = self.draw_buy_bar(current_image, ask_bid_lines, ask_bid_lines, self.position , self.current_trade["profit"], self.prev_trade )
        current_image_m5 = self.draw_buy_bar(current_image_m5, ask_bid_lines_m5, ask_bid_lines_m5, self.position, self.current_trade["profit"], self.prev_trade)
        self.trade_queue.append(ask_bid_lines)
        self.trade_queue_m5.append(ask_bid_lines_m5)
        self.image_queue.append(current_image)  
        self.image_queue_m5.append(current_image_m5)
        
        if self.env_draw:
            self.draw_collection(self.image_queue) 
            self.draw_collection(self.image_queue_m5)
        self.reward = 0#self.account_balance/self.account_balance
        
    def enter_idle_mode(self, ask_bid_lines, ask_bid_lines_m5, current_image, current_image_m5  ):
        
        self.current_trade["ask_bid_lines"] = ask_bid_lines
        self.current_trade["ask_bid_lines_m5"] = ask_bid_lines_m5 
        self.current_trade["entry_price"] = float(f"{self.m1short.iloc[self.current_step]['Ask']}")
        self.current_trade["current_price"] = float(f"{self.m1short.iloc[self.current_step]['Ask']}")
        self.current_trade["position"] = self.position
        self.current_trade["reward"] = 0
        self.current_trade["timesteps"] += 1
        self.current_trade["profit"] = 0
        self.current_trade["parent_ask_bid_lines"] = None
        self.current_trade["parent_ask_bid_lines_m5"] = None
        self.current_trade["balance"] = self.account_balance
        current_image = self.draw_buy_bar(current_image, ask_bid_lines, ask_bid_lines, self.position, self.current_trade["profit"], self.prev_trade)
        current_image_m5 = self.draw_buy_bar(current_image_m5, ask_bid_lines_m5, ask_bid_lines_m5, self.position, self.current_trade["profit"], self.prev_trade)
        self.trade_queue.append(ask_bid_lines)
        self.trade_queue_m5.append(ask_bid_lines_m5)
        self.image_queue.append(current_image)  
        self.image_queue_m5.append(current_image_m5)
        if self.env_draw:
            self.draw_collection(self.image_queue) 
            self.draw_collection(self.image_queue_m5)
        self.reward = 0
        
    def hold_position(self,ask_bid_lines, ask_bid_lines_m5, current_image, current_image_m5):
        
        self.current_trade["current_price"] = float(f"{self.m1short.iloc[self.current_step]['Bid']}") if self.position == "buy" else float(f"{self.m1short.iloc[self.current_step]['Ask']}")
        self.current_trade["profit"] = self.current_trade["current_price"] - self.current_trade["entry_price"] if self.position == "buy" else self.current_trade["entry_price"] - self.current_trade["current_price"]
        self.current_trade["timesteps"] += 1
        self.current_trade["balance"] = self.account_balance
        self.current_trade["highest"] = self.current_trade["profit"] if self.current_trade["profit"] > self.current_trade["highest"] else self.current_trade["highest"]
        #self.current_trade["highest"] -= 15

        current_image = self.draw_buy_bar(current_image, self.current_trade["parent_ask_bid_lines"], ask_bid_lines, self.position , self.current_trade["profit"], self.prev_trade)
        current_image_m5 = self.draw_buy_bar(current_image_m5, self.current_trade["parent_ask_bid_lines_m5"], ask_bid_lines_m5, self.position, self.current_trade["profit"], self.prev_trade)
        self.trade_queue.append(ask_bid_lines)
        self.trade_queue_m5.append(ask_bid_lines_m5)
        self.image_queue.append(current_image)
        self.image_queue_m5.append(current_image_m5)
        if self.env_draw:
            self.draw_collection(self.image_queue) 
            self.draw_collection(self.image_queue_m5)
        if self.current_trade["profit"] >= 0:
            self.reward = self.get_reward(self.current_trade)#(self.account_balance + self.current_trade["profit"])/self.account_balance
            self.current_trade["reward"] = self.reward
        else:
            self.reward = self.get_reward(self.current_trade)#-1 - (self.account_balance + self.current_trade["profit"])/self.account_balance
            self.current_trade["reward"] = self.reward

        
                
    def close_position(self, ask_bid_lines, ask_bid_lines_m5, current_image, current_image_m5, prev_trade = None):
        # print(f"self.position {self.position}")
        self.current_trade["current_price"] = float(f"{self.m1short.iloc[self.current_step]['Bid']}") if self.position == "buy" else float(f"{self.m1short.iloc[self.current_step]['Ask']}")
        self.current_trade["profit"] = self.current_trade["current_price"] - self.current_trade["entry_price"] if self.position == "buy" else self.current_trade["entry_price"] - self.current_trade["current_price"]
        self.current_trade["highest"] = self.current_trade["profit"] if self.current_trade["profit"] > self.current_trade["highest"] else self.current_trade["highest"]
        #self.current_trade["highest"] += 5

        current_image = self.draw_buy_bar(current_image, self.parent_ask_bid_lines, ask_bid_lines, self.position, self.current_trade["profit"] , prev_trade)
        current_image_m5 = self.draw_buy_bar(current_image_m5, self.parent_ask_bid_lines_m5, ask_bid_lines_m5, self.position, self.current_trade["profit"], prev_trade)
        self.trade_queue.append(ask_bid_lines)
        self.trade_queue_m5.append(ask_bid_lines_m5)
        self.image_queue.append(current_image)
        self.image_queue_m5.append(current_image_m5)
        self.account_balance += self.current_trade["profit"]
        if self.env_draw:
            self.draw_collection(self.image_queue) 
            self.draw_collection(self.image_queue_m5)
        if self.current_trade["profit"] >= 0:
            self.reward = self.get_reward(self.current_trade)##(self.account_balance + self.current_trade["profit"])/self.account_balance
        else:
            self.reward =  self.get_reward(self.current_trade)#-1 - (self.account_balance + self.current_trade["profit"])/self.account_balance
        #self.trade_queue, self.image_queue = self.reset_collection(self.trade_queue,self.image_queue)
        # self.account_balance += self.current_trade["profit"]
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
        self.action = action
        
        ask_bid_lines, current_image = self.get_ask_bid_lines(f"{self.m1short.iloc[self.current_step]['image_path']}")
        ask_bid_lines_m5, current_image_m5 = self.get_ask_bid_lines(f"{self.m5short.iloc[self.current_step]['image_path']}")
        ask_bid_lines["key"] = self.current_step
        ask_bid_lines_m5["key"] = self.current_step
        
        if self.position == "close":
            if action == 1:
                self.position = "buy"
                self.prev_trade.append(1)
                self.parent_ask_bid_lines = ask_bid_lines
                self.parent_ask_bid_lines_m5 = ask_bid_lines_m5
                
                self.enter_trade(ask_bid_lines, ask_bid_lines_m5,current_image, current_image_m5)
            elif action == 2:
                self.position = "sell"
                self.prev_trade.append(2)
                self.parent_ask_bid_lines = ask_bid_lines
                self.parent_ask_bid_lines_m5 = ask_bid_lines_m5
                self.enter_trade(ask_bid_lines, ask_bid_lines_m5, current_image, current_image_m5)
            elif action == 0:
                self.position = "close"
                self.prev_trade.append(0)
                self.parent_ask_bid_lines = ask_bid_lines
                self.parent_ask_bid_lines_m5 = ask_bid_lines_m5
                self.enter_idle_mode(ask_bid_lines, ask_bid_lines_m5, current_image, current_image_m5)
        elif self.position == "buy":
            if action == 1:
                self.prev_trade.append(1)
                self.hold_position(ask_bid_lines, ask_bid_lines_m5, current_image, current_image_m5)
            elif action in [2]:
                self.prev_trade.append(0)
                self.close_position(ask_bid_lines, ask_bid_lines_m5, current_image, current_image_m5, self.prev_trade)                
                #fix standard limit and standard balance on chart 
                if self.account_balance > int((2/3) * self.chart_decorator.balance_limit) + self.chart_decorator.standard_balance:
                    # print("\n\nReseting the account balance in closing buy trade using 2\n\n the if statement\n\n\n")
                    self.chart_decorator.standard_balance = self.account_balance
                # elif self.chart_decorator.standard_balance * (2/3) > self.account_balance :
                    # print("\n\nReseting the account balance in closing buy trade using 2\n\n the elif statement\n\n\n")
                    # self.chart_decorator.standard_balance -= self.chart_decorator.balance_limit  
                # self.reward = (self.account_balance + self.current_trade["profit"])/1000
                # self.current_trade["reward"] = (self.account_balance + self.current_trade["profit"])/1000
                    
            elif action in [0]:
                self.prev_trade.append(0)
                self.close_position(ask_bid_lines, ask_bid_lines_m5, current_image, current_image_m5, self.prev_trade)   
                
                #fix standard limit and standard balance on chart
                if self.account_balance > int((2/3) * self.chart_decorator.balance_limit) + self.chart_decorator.standard_balance:
                    # print("\n\nReseting the account balance in closing buy trade using 0\n\n the if statement\n\n\n")
                    self.chart_decorator.standard_balance = self.account_balance
                # elif self.chart_decorator.standard_balance * (2/3) > self.account_balance :
                    # print("\n\nReseting the account balance in closing buy trade using 0\n\n the elif statement\n\n\n")
                    # self.chart_decorator.standard_balance -= self.chart_decorator.balance_limit  
            
        elif self.position == "sell":
            if action == 2:
                self.prev_trade.append(2)
                self.hold_position(ask_bid_lines, ask_bid_lines_m5, current_image, current_image_m5)
            #this reward is to penalize it not to change the position without closing the position
            elif action in [1]:
                self.prev_trade.append(0)
                self.close_position(ask_bid_lines, ask_bid_lines_m5, current_image, current_image_m5, self.prev_trade)
                #fix standard limit and standard balance on chart
                if self.account_balance > int((2/3) * self.chart_decorator.balance_limit) + self.chart_decorator.standard_balance:
                    # print("\n\nReseting the account balance in closing sell trade using 1\n\n the if statement\n\n\n")
                    
                    self.chart_decorator.standard_balance = self.account_balance
                # elif self.chart_decorator.standard_balance * (2/3) > self.account_balance :
                    # print("\n\nReseting the account balance in closing sell trade using 1\n\n the elif statement\n\n\n")
                    # self.chart_decorator.standard_balance -= self.chart_decorator.balance_limit  
                # self.reward = (self.account_balance + self.current_trade["profit"])/1000
                # self.current_trade["reward"] = (self.account_balance + self.current_trade["profit"])/1000
            elif action == 0:
                self.prev_trade.append(0)
                self.close_position(ask_bid_lines, ask_bid_lines_m5, current_image, current_image_m5, self.prev_trade)
                # self.prev_trade.append(0)
                #fix standard limit and standard balance on chart
                if self.account_balance > int((2/3) * self.chart_decorator.balance_limit) + self.chart_decorator.standard_balance:
                    # print("\n\nReseting the account balance in closing sell trade using 0\n\n the if statement\n\n\n")
                    # print(f"Account Balance == {self.account_balance} chart decorator standard_balance {self.chart_decorator.standard_balance}")
                    self.chart_decorator.standard_balance = self.account_balance
                    # print(f"updated standard balance == {self.chart_decorator.standard_balance}")
                # elif self.chart_decorator.standard_balance * (2/3) > self.account_balance :
                    # print("\n\nReseting the account balance in closing sell trade using 0\n\n the elif statement\n\n\n")
                    # self.chart_decorator.standard_balance -= self.chart_decorator.balance_limit 
        self.current_step+=1
        next_state = self.get_obs()
        if self.account_balance < 810:
            self.done = True
        # print(self.trade_queue)
        return next_state, self.reward, self.done, self.current_trade
    
    def get_obs(self):
        stacked_obs =   np.expand_dims(np.stack([
                        np.asarray(self.image_queue_m5[0])/255, 
                        np.asarray(self.image_queue_m5[1])/255, 
                        np.asarray(self.image_queue_m5[2])/255, 
                        np.asarray(self.image_queue_m5[3])/255,
                        np.asarray(self.image_queue[0])/255, 
                        np.asarray(self.image_queue[1])/255, 
                        np.asarray(self.image_queue[2])/255, 
                        np.asarray(self.image_queue[3])/255
            ]), axis=0)
        # stacked_obs_m5 = np.expand_dims(np.stack([np.asarray(self.image_queue_m5[0])/255, np.asarray(self.image_queue_m5[1])/255, np.asarray(self.image_queue_m5[2])/255, np.asarray(self.image_queue_m5[3])/255]), axis=0)
        # stacked = np.concatenate([stacked_obs_m5, stacked_obs], axis=0)
        return np.expand_dims(stacked_obs, axis=0)
    
    
    
    
class TimeSformerBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_heads):
        super(TimeSformerBlock, self).__init__()
        
        # Define the self-attention layer
        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)
        
        # Define the feedforward layer
        self.feedforward = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim * 4, activation='relu'),
            tf.keras.layers.Dense(hidden_dim)
        ])
        
        # Define the layer normalization layers
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        
    def call(self, x):
        # Apply layer normalization and self-attention
        norm_x = self.norm1(x)
        attention_output = self.self_attention(norm_x, norm_x)
        x = x + attention_output
        
        # Apply layer normalization and feedforward layer
        norm_x = self.norm2(x)
        feedforward_output = self.feedforward(norm_x)
        x = x + feedforward_output
        
        return x
    
    
    
class Agent:
    def __init__(self, state_size, action_size, env, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 0.001
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.gamma = 0.99
        self.env = env
        self.actor = self._actor_model()
        self.critic = self._critic_model()
        # self.critic.set_weights(self.actor.get_weights())
        self.models_dir = "models/attention/"
        self.actor_model_name = "T32_v4R_att_actor.h5"
        self.critic_model_name = "T32_v4R_att_critic.h5"
        self.actor.summary()
        
        
    def _actor_model(self):
        cnn = Sequential()
        cnn.add(tf.keras.layers.Conv3D(32, (3,3,3), strides=(1,4,4), padding="same"))
        cnn.add(Activation('relu'))
        cnn.add(tf.keras.layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same'))
        cnn.add(tf.keras.layers.Conv3D(64, (3,3,3), padding="same"))
        cnn.add(Activation('relu'))
        cnn.add(tf.keras.layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same'))
        cnn.add(tf.keras.layers.Conv3D(64, (3,3,3), padding="same"))
        cnn.add(Activation('relu'))
        cnn.add(Flatten())
        cnn.add(Dense(512, activation='relu'))
        transformer = TimeSformerBlock( num_heads=8, hidden_dim=512)
        model=Sequential()
        model.add(tf.keras.layers.TimeDistributed(cnn,input_shape=self.state_size))
        model.add(transformer)
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=1e-4))
        return model
    
    def _critic_model(self):
        cnn = Sequential()
        cnn.add(tf.keras.layers.Conv3D(32, (3,3,3), strides=(1,4,4), padding="same"))
        cnn.add(Activation('relu'))
        cnn.add(tf.keras.layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same'))
        cnn.add(tf.keras.layers.Conv3D(64, (3,3,3), padding="same"))
        cnn.add(Activation('relu'))
        cnn.add(tf.keras.layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same'))
        cnn.add(tf.keras.layers.Conv3D(64, (3,3,3), padding="same"))
        cnn.add(Activation('relu'))
        cnn.add(Flatten())
        cnn.add(Dense(512, activation='relu'))
        transformer = TimeSformerBlock( num_heads=8, hidden_dim=512)
        model=Sequential()
        model.add(tf.keras.layers.TimeDistributed(cnn,input_shape=self.state_size))
        model.add(transformer)
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(learning_rate=1e-4))
        return model
    
    def predict(self, state):
        return self.actor.predict(state)
        
    def save(self):
        self.actor.save(f"{self.models_dir}{self.actor_model_name}")
        self.critic.save(f"{self.models_dir}{self.critic_model_name}")
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return secrets.randbelow(self.action_size) if np.random.choice(20) % 2 == 0 else np.random.choice(self.action_size) , [[0,0,0]]
        else:
            pred = self.actor.predict(state)
            return np.argmax(pred[0][0]), pred 
        
    def load(self):
        self.actor.load_weights(f"{self.models_dir}{self.actor_model_name}")
        self.critic.load_weights(f"{self.models_dir}{self.critic_model_name}")
        
    def test_agent(self, draw=False):
        header = ["step","action", "reward", "done","profit", "Episode Profit", "balance", "Eprice", "Cprice", ]
        state, reward, done, current_trade = self.env.reset(800)
        profits = 0
        clear_output(wait=True)    
        chart_decorator = ChartDecorator()
        env = ForexCustomEnv(m1short,m5short, chart_decorator)
        self.env = env
        self.env.env_draw = draw
        # agent = Agent(state_size, action, env)
        self.env.account_balance = 1000
        self.epsilon = 0.001#1.0   
        state, reward, done, _ = self.env.reset(4)
      
        for step in range(4, 1200, 1):
            rows = []
            action, pred = self.act(state)
            value  = self.critic.predict(state)[0]
            
            next_state, reward, done, current_trade = self.env.step(action)
            if action == 0:
                profits += current_trade["profit"]
            row = [step, action, reward,done, current_trade["profit"], profits , current_trade["balance"], current_trade["entry_price"], current_trade["current_price"]]
            rows.append(row)
            print(tabulate(rows, headers=header, tablefmt='grid'))
            print(f"actor {pred} critic {value}")
# print(f"current_trade {current_trade}\n")
            state = next_state
            # profits.append(current_trade["profit"])
            

    def train(self, num_episodes, draw = False, epsilon = 0.001):
        #here we train the network on all zeros
        #we train it on penalties by greedy exporation
        #we train it on expert advise 
        #we let the network train on itself finally and evaluate it 
        header = ["episode","epsilon", "counter","action", "done" ,"profit", "balance","Eprice","Cprice", "critic target_f_values","target_f__reward", "actor pred qvls", "fit actor q_values", "total_profits"]
        self.episode_rewards = []
        target_f_values = []
        for episode in range(num_episodes):         
            clear_output(wait=True)    
            chart_decorator = ChartDecorator()
            env = ForexCustomEnv(m1short,m5short, chart_decorator)
            self.env = env
            # agent = Agent(state_size, action, env)
            self.env.account_balance = 1000
            self.epsilon = epsilon
            self.env.env_draw = draw
            state, reward, done, _ = self.env.reset()
            total_rewards = 0
            counter = 0
            total_profits = 0
            while not done:
                rows = []
                action, pred = self.act(state)
                next_state, reward, done, current_trade = self.env.step(action)
                total_rewards += reward
                total_profits += current_trade["profit"]
                if not done:
                    target_f_values = self.critic.predict(next_state)
                    target_f = reward + self.gamma * target_f_values[0][0]
                else:
                    target_f = reward
                q_values = self.actor.predict(state)
                try:
                    q_values_copy = np.array(q_values, copy=True)
                    q_values_copy[0][0][action] = target_f
                    row = [episode,self.epsilon,counter, action, done,  current_trade["profit"], current_trade["balance"],current_trade["entry_price"], current_trade["current_price"], target_f_values[0][0], target_f, ",".join(str("{:.2f}".format(x)) for x in list(q_values)[0][0]),",".join(str("{:.2f}".format(x)) for x in list(q_values_copy)[0][0]), total_profits]
                    rows.append(row)
                    print(tabulate(rows,headers = header, tablefmt="grid"))
                except Exception as e:
                    print(f"Exception {e}")
                self.critic.fit(state, np.array(q_values[0][0][action]).reshape(1,1,1) , verbose=0)
                q_values[0][0][action] = target_f
                self.actor.fit(state, q_values, verbose=0)
                state = next_state
                if counter % 100 == 0 and counter > 0:
                    self.save()
                # Decay the exploration rate
                if self.epsilon > self.epsilon_min and counter % 20 == 0:
                    self.epsilon *= self.epsilon_decay
                if counter > 1200:
                    done = True
                counter += 1
            clear_output(wait=True)
            if episode % 3 == 0:
                self.save()
            # Decay the exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.episode_rewards.append(total_rewards)
            # print(self.episode_rewards)
        return self.episode_rewards

    def expert_choices(self, index):
        choices = list(range(12, 23,1))
        choice2 = list(range(25, 38 , 1))
        choice3 = list(range(5,11,1))
        choice4 = list(range(39,45, 1))
        choice5 = list(range(46, 52,1  ))
        
        choice6 = list(range(53,69,1)) #2
        choice7 = list(range(70, 79 , 1)) #1
        choice8 = list(range(80, 103, 1)) #2
        choice9 = list(range(104, 114, 1)) # 1
        choice10 = list(range(115, 119, 1)) # 2
        choice11 = list(range(125, 128, 1)) # 2
        choice12 = list(range(133, 140, 1)) # 2
        choice13 = list(range(141, 146, 1)) # 1
        choice14 = list(range(147, 163, 1)) #2
        choice15 = list(range(164, 173, 1)) # 1
        choice16 = list(range(174, 192, 1)) #2    
        choice17 = list(range(193, 204, 1)) #1
        choice18 = list(range(205,211 , 1)) #2
        choice19 = list(range(216,225,1)) #1
        choice20 = list(range(226, 232, 1)) #2
        choice21 = list(range(237, 254, 1)) #2
        choice22 = list(range(259, 305, 1)) #1
        choice23 = list(range(305, 315, 1)) #2
        choice24 = list(range(316, 337, 1)) #1
        choice25 = list(range(338,372, 1)) #2
        choice26 = list(range(373, 435, 1)) #1
        choice27 = list(range(437, 467, 1)) # 2
        choice28 = list(range(468, 482, 1)) # 1
        choice29 = list(range(483, 493 , 1)) # 2
        choice30 = list(range(494, 498, 1)) # 1
        choice31 = list(range(499, 518, 1)) # 2
        choice32 = list(range(519, 525, 1)) # 1 buy
        choice33 = list(range(530, 536, 1)) # 2
        choice34 = list(range(536, 544, 1)) # 1        
        
        
        action = 0

        if index in choices:
            action = 2 
            # agent.env.env_draw = False
        if index in choice2:
            action = 1
            # agent.env.env_draw = True
        # if index == 5:
        #     print(f"index == 5, index == {index}")
        if index in choice3:
            action = 1
        if index in choice4:
            action = 2
        if index in choice5:
            action = 1
        if index in choice6:
            action = 2
        if index in choice7:
            action = 1
        if index in choice8:
            action = 2
        if index in choice9:
            action = 1
        if index in choice10:
            action = 2
        if index in choice11:
            action = 2
        if index in choice12:
            action = 2
        if index in choice13:
            action = 1
        if index in choice14:
            action = 2
            
        if index in choice15:
            action = 1
        if index in choice16:
            action = 2            
        if index in choice17:
            action = 1   
        if index in choice18:
            action = 2    
        if index in choice19:
            action = 1             
        if index in choice20:
            action = 2    
        if index in choice21:
            action = 2
        if index in choice22:
            action = 1
        if index in choice23:
            action = 2
        if index in choice24:
            action = 1  
        if index in choice25:
            action = 2
        if index in choice26:
            action = 1
        if index in choice27:
            action = 2 
        if index in choice28:
            action = 1   
        if index in choice29:
            action = 2      
        if index in choice30:
            action = 1
        if index in choice31:
            action = 2
        if index in choice32:
            action = 1
        if index in choice33:
            action = 2
        if index in choice34:
            action = 1
            
        return action   
    def train_expert_advice(self):
        header = ["episode","epsilon","action", "done" ,"profit", "balance","Eprice","Cprice", "critic target_f_values","target_f__reward", "actor pred qvls", "fit actor q_values", "total_profits"]
        clear_output(wait=True)    
        chart_decorator = ChartDecorator()
        env = ForexCustomEnv(m1short,m5short, chart_decorator)
        self.env = env
        # agent = Agent(state_size, action, env)
        self.env.account_balance = 1000
        epsilon = self.epsilon
        self.epsilon = 0.001
        self.env.env_draw = False
                # state, reward, done, _ = self.env.reset()
        total_rewards = 0
        counter = 0
        total_profits = 0        
        
        state, reward, done, _ = self.env.reset(4)
        total_profits = 0
        for episode in range(4,50,1):
            rows = []
            action = self.expert_choices(episode)#self.act(state)
            next_state, reward, done, current_trade = self.env.step(action)
            
            if not done:
                target_f_values = self.critic.predict(next_state)
                target_f = reward + self.gamma * target_f_values[0][0]
            else:
                target_f = reward
            q_values = self.actor.predict(state)
            try:
                q_values_copy = np.array(q_values, copy=True)
                q_values_copy[0][0][action] = target_f
                row = [episode,self.epsilon, action, done,  current_trade["profit"], current_trade["balance"],current_trade["entry_price"], current_trade["current_price"], target_f_values[0][0], target_f, ",".join(str("{:.2f}".format(x)) for x in list(q_values)[0][0]),",".join(str("{:.2f}".format(x)) for x in list(q_values_copy)[0][0]), total_profits]
                rows.append(row)
                print(tabulate(rows,headers = header, tablefmt="grid"))
            except Exception as e:
                print(f"Exception {e}")
                
            self.critic.fit(state, np.array(q_values[0][0][action]).reshape(1,1,1) , verbose=2)
            
            q_values[0][0][action] = target_f
            
            self.actor.fit(state, q_values, verbose=2)
            
            state = next_state             
            
            if episode % 100 == 0:
                self.save()
        self.epsilon = epsilon

    def train_on_zeros(self):
        header = ["episode","epsilon","action", "done" ,"profit", "balance","Eprice","Cprice", "critic target_f_values","target_f__reward", "actor pred qvls", "fit actor q_values", "total_profits"]
        state, reward, done, _ = self.env.reset(4)
        total_profits = 0
        for episode in range(4,546,1):
            rows = []
            action = 0#self.expert_choices(episode)#self.act(state)
            next_state, reward, done, current_trade = self.env.step(action)
            
            if not done:
                target_f_values = self.critic.predict(next_state)
                target_f = reward + self.gamma * target_f_values[0][0]
            else:
                target_f = reward
            q_values = self.actor.predict(state)
            # try:
            #     q_values_copy = np.array(q_values, copy=True)
            #     q_values_copy[0][0][action] = target_f
            #     row = [episode,self.epsilon, action, done,  current_trade["profit"], current_trade["balance"],current_trade["entry_price"], current_trade["current_price"], target_f_values[0][0], target_f, ",".join(str("{:.2f}".format(x)) for x in list(q_values)[0][0]),",".join(str("{:.2f}".format(x)) for x in list(q_values_copy)[0][0]), total_profits]
            #     rows.append(row)
            #     # print(tabulate(rows,headers = header, tablefmt="grid"))
            # except Exception as e:
                # print(f"Exception {e}")
                
            self.critic.fit(state, np.array(q_values[0][0][action]).reshape(1,1,1) , verbose=2)
            
            q_values[0][0][action] = target_f
            
            self.actor.fit(state, q_values, verbose=2)
            
            state = next_state             
            
            # if episode % 50 == 0:
            #     self.save()
            
            
            
# m5long = pd.read_csv("final_m5long.csv")
m5short = pd.read_csv("final_m5short.csv")
# m1long = pd.read_csv("final_m1long.csv")
# m1short = pd.read_csv("final_complete_m1short.csv")
m1short = pd.read_csv("final_m1short.csv")

action = 3
state_size = (1, 8, 224, 224, 3)
chart_decorator = ChartDecorator()
env = ForexCustomEnv(m1short,m5short, chart_decorator)
# env.env_draw = True
agent = Agent(state_size, action, env)
agent.env = env
agent.epsilon = 0.001
#agent.train_expert_advice()

#agent.epsilon = 0.7
# agent.env.env_draw = True
agent.env.account_balance = 1000
agent.epsilon = 0.001

# agent.train(2)
agent.load()


e = 0.5

# for step in range(10):
# agent.train_on_zeros()
for _ in range(20):
    print(f"Episode {_}")
    # try:
    #     agent.test_agent()
    # except:
    #     pass
    # for __ in range(1):
    #agent.train_expert_advice()
    #clear_output(wait=True)    
    #agent.train(6, False, agent.epsilon)
    # clear_output(wait=True) 
    # agent.train_expert_advice()
    # clear_output(wait=True)
    #agent.train_expert_advice()
    #clear_output(wait=True)
    # agent.train(1)
    # clear_output(wait=True) 
    agent.test_agent()
