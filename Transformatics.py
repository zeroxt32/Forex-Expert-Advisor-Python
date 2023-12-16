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




class ChartDecorator:
    def __init__(self):
        self.balance_limit = 300
        self.standard_balance = 1000
        
    def reset_chart(self, img, draw_context):
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
            draw_context.rectangle([x1, y1, x2, y2], fill=(71, 44, 27)) 

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

    def draw_profit_bar(self, img, draw_context, profit = 0, account_balance = 1000):
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

            account_balance_bar = (200*account_balance)/2000
            x1, y1 = 0, 10
            x2, y2 = account_balance_bar, 14
            draw_context.rectangle([x1, y1, x2, y2], fill=(255, 119, 0)) 


    def draw_buy_bar(self, img2, start_lines, end_lines, position = None, profit = 0, prev_trade=None, account_balance = 1100):

        from PIL import Image, ImageDraw
        d = ImageDraw.Draw(img2)
        
        if start_lines["ask_line"] == None:
            start_lines["ask_line"] = start_lines["bid_line"] + 2
        elif start_lines["bid_line"] == None:
            start_lines["bid_line"] == start_lines["ask_line"] - 2
        elif end_lines["ask_line"] == None:
            end_lines["ask_line"] = end_lines["bid_line"] + 2
        elif end_lines["bid_line"] == None:
            end_lines["bid_line"] == end_lines["ask_line"] - 2
        
        self.reset_chart(img2, d)
        self.draw_account_balance(img2, d, account_balance)

        if position == "sell":
            #draw icon for trade setup
            self.draw_current_trade(img2, d, "sell")
            # Draw indicator of trade start
            line_color = (200,200,200)
            top = (200, end_lines["ask_line"])
            bottom = (200, end_lines["bid_line"])
            d.line([top, bottom], fill=line_color, width=20)

            #Draw start marker and line
            line_color = (0, 255, 0)
            top = (200, start_lines["ask_line"])
            bottom = (200, start_lines["bid_line"])
            d.line([top, bottom], fill=line_color, width=30) 
            #Draw start line
            line_color = (0, 255, 0)#(252, 81, 48)
            top = (50, start_lines["ask_line"])
            bottom = (50, start_lines["bid_line"])
            d.line([top, bottom], fill=line_color, width=250) 

            #Draw profit indicator of sell
            if profit < 0:
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
        elif position == "buy":
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

        elif position == "close" and prev_trade == None:
            self.draw_current_trade(img2, d, "close")
            
            line_color = (255,0,255)
            top = (200, end_lines["ask_line"])
            bottom = (200, end_lines["bid_line"])
            d.line([top, bottom], fill=line_color, width=30)  

            self.draw_profit_bar(img2, d, profit)
        else:
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

            self.draw_profit_bar(img2, d, profit)

            if prev_trade == 2 and profit > 0:   
                #print right bar for the trade that has closed
                line_color = (0, 255, 0)
                top = (213, start_lines["ask_line"])
                bottom = (213, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10) 

                line_color = (241, 254, 198)
                top = (190, start_lines["ask_line"])
                bottom = (190, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10)             

            elif prev_trade == 2 and profit < 0:
                line_color = (0, 255, 0)
                top = (213, start_lines["ask_line"])
                bottom = (213, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10) 

                line_color = (255,0,0)
                top = (190, start_lines["ask_line"])
                bottom = (190, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10)    
            elif prev_trade == 1 and profit > 0:
                line_color = (255, 255, 0)
                top = (213, start_lines["ask_line"])
                bottom = (213, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10) 

                line_color = (241, 254, 198)
                top = (190, start_lines["ask_line"])
                bottom = (190, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10)    

            elif prev_trade == 1 and profit < 0:
                line_color = (255, 255, 0)
                top = (213, start_lines["ask_line"])
                bottom = (213, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10) 

                line_color = (255,0,0)
                top = (190, start_lines["ask_line"])
                bottom = (190, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10) 

        return img2

    
class ForexCustomEnv:
    def __init__(self, m1short, chart_decorator):
        self.m1short = m1short
        self.trade_queue = deque(maxlen=4)
        self.image_queue = deque(maxlen=4)
        self.dataset_directory = "colabM1M5/episode2/"
        self.account_balance = 1000
        self.env_draw = False
        self.chart_decorator = chart_decorator
        
        self.prev_trade = deque(maxlen=4)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        
        

    def get_ask_bid_lines(self, image_path):
        ask_bid_lines = {"ask_line":None, "bid_line":None, "key": 0}
        img = Image.open(image_path)
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
                
        return ask_bid_lines , img          
    
    def reset(self, current_step = 4, key1 = 0, key2 = 1, key3 = 2, key4 = 3, slot1 = 4, slot2 = 3, slot3 = 2, slot4 = 1):
        self.current_step = 4
        linesq, imageq = self.get_ask_bid_lines(f"{self.dataset_directory}{self.m1short.iloc[self.current_step-slot1]['image_path']}")
        linesq["key"] = key1
        linest, imaget = self.get_ask_bid_lines(f"{self.dataset_directory}{self.m1short.iloc[self.current_step-slot2]['image_path']}")
        linest["key"] = key2
        liness, images = self.get_ask_bid_lines(f"{self.dataset_directory}{self.m1short.iloc[self.current_step-slot3]['image_path']}")
        liness["key"] = key3
        linesc, imagec = self.get_ask_bid_lines(f"{self.dataset_directory}{self.m1short.iloc[self.current_step-slot4]['image_path']}")
        linesc["key"] = key4
        self.trade_queue.append(linesq)
        self.trade_queue.append(linest)
        self.trade_queue.append(liness)
        self.trade_queue.append(linesc)
        self.image_queue.append(imageq)
        self.image_queue.append(imaget)
        self.image_queue.append(images)
        self.image_queue.append(imagec)    
        self.done = False
        self.reward = 0
        self.position = None
        self.reset_current_trade()
        self.account_balance = 1000
        next_state = self.get_obs()
        self.episode_profit = 0
        self.prev_trade = deque(maxlen=4)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        self.prev_trade.append(0)
        
        #reset standard limit in chart decorator
        self.chart_decorator.standard_limit = self.account_balance
        
        return next_state, self.reward, self.done, self.current_trade 
        
    def reset_current_trade(self):
        self.current_trade = {}
        self.current_trade["current_step"] = self.current_step
        self.current_trade["ask_bid_lines"] = None
        self.current_trade["entry_price"] = None
        self.current_trade["current_price"] = None
        self.current_trade["timesteps"] = 0
        self.current_trade["position"] = None
        self.current_trade["profit"] = 0
        self.current_trade["position"] = None
        self.current_trade["parent_ask_bid_lines"] = None
        self.current_trade["reward"] = 0
        self.current_trade["balance"] = self.account_balance
        self.current_trade["episode_profit"] = 0
        
        
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
        
    def draw_buy_bar(self, current_image, start_lines, end_lines, position = None, profit = 0):
        # def draw_buy_bar(self, img2, start_lines, end_lines, position = None, profit = 0, prev_trade=None, account_balance = 1100):
        return self.chart_decorator.draw_buy_bar(current_image, start_lines, end_lines, self.position, profit, self.prev_trade[2], self.account_balance)
#         try:
#             if start_lines["ask_line"] == None:
#                 start_lines["ask_line"] = start_lines["bid_line"] + 2
#             elif start_lines["bid_line"] == None:
#                 start_lines["bid_line"] == start_lines["ask_line"] - 2
#             elif end_lines["ask_line"] == None:
#                 end_lines["ask_line"] = end_lines["bid_line"] + 2
#             elif end_lines["bid_line"] == None:
#                 end_lines["bid_line"] == end_lines["ask_line"] - 2
                
#             from PIL import Image, ImageDraw
#             d = ImageDraw.Draw(current_image)
#             if self.position == "sell":
#                 # Draw indicator of trade start
#                 line_color = (200,200,200)
#                 top = (200, end_lines["ask_line"])
#                 bottom = (200, end_lines["bid_line"])

#                 d.line([top, bottom], fill=line_color, width=20)
#             #Draw profit indicator of sell
#                 if profit < 0:
#                     line_color = (0, 0, 255)
#                     top = (200, start_lines["ask_line"])
#                     bottom = (200,end_lines["bid_line"])
#                     d.line([top, bottom], fill=line_color, width=10)
#                 else:
#                     line_color = (0, 255, 0)
#                     top = (200, start_lines["ask_line"])
#                     bottom = (200,end_lines["bid_line"])
#                     d.line([top, bottom], fill=line_color, width=10)
#             #Draw loss indicator for sell
#             elif self.position == "buy":
#                 #Draw indicator of trade start
#                 line_color = (0,255,0)
#                 top = (200, end_lines["ask_line"])
#                 bottom = (200, end_lines["bid_line"])
#                 d.line([top, bottom], fill=line_color, width=20)        
#                 if profit < 0:
#                     line_color = (255, 0, 0)
#                     top = (200, start_lines["ask_line"])
#                     bottom = (200, end_lines["bid_line"])
#                     d.line([top, bottom], fill=line_color, width=10)
#                 else:
#                     line_color = (255, 255, 0)
#                     top = (200, start_lines["ask_line"])
#                     bottom = (200, end_lines["bid_line"])
#                     d.line([top, bottom], fill=line_color, width=10)
#             else:
#                 line_color = (255,0,255)
#                 top = (200, start_lines["ask_line"])
#                 bottom = (200, end_lines["bid_line"])
#                 d.line([top, bottom], fill=line_color, width=40)
#         except Exception as e:
#             print(f"\n\n\nstart_lines: {start_lines} end_lines:{end_lines}{e}\n\n\n")
#         return current_image
    
    '''
    @description: Enters a trade in the forex environment
                  Enters the trade in the current time step, sets the position of the trade. sets the ask bid lines for the current frame
                  sets the entry price for the trade, and profit = 0. Then draws on the image the entry icon on the chart
                  and saves the updated current_image with modifications to the image queue with its corresponding ask and bid lines. 
                  then initializes the timesteps of the current trade to 0. sets the reward to 0.3
    @params
                  @position = None the position of the current_trade
                  @ask_bid_lines of the current_image
                  @current_image the current frame of the environment
                  
    '''
    
    def enter_trade(self, ask_bid_lines,current_image):
        self.current_trade["position"] = self.position
        self.current_trade["current_step"] = self.current_step
        self.current_trade["ask_bid_lines"] = ask_bid_lines
        self.current_trade["entry_price"] = float(f"{self.m1short.iloc[self.current_step]['Ask']}") if self.position == "buy" else float(f"{self.m1short.iloc[self.current_step]['Bid']}")
        self.current_trade["profit"] = 0
        self.current_trade["current_price"] = float(f"{self.m1short.iloc[self.current_step]['Ask']}")
        self.current_trade["timesteps"] = 0
        self.current_trade["reward"] = 0.4
        self.current_trade["parent_ask_bid_lines"] = ask_bid_lines
        self.current_trade["balance"] = self.account_balance
        current_image = self.draw_buy_bar(current_image, ask_bid_lines, ask_bid_lines, self.position , self.current_trade["profit"])
        self.trade_queue.append(ask_bid_lines)
        self.image_queue.append(current_image)  
        if self.env_draw:
            self.draw_collection(self.image_queue) 
        self.reward = 0.4
        
    def enter_idle_mode(self, ask_bid_lines, current_image ):
        self.current_trade["current_step"] = self.current_step
        self.current_trade["ask_bid_lines"] = ask_bid_lines
        self.current_trade["entry_price"] = float(f"{self.m1short.iloc[self.current_step]['Ask']}")
        self.current_trade["current_price"] = float(f"{self.m1short.iloc[self.current_step]['Ask']}")
        self.current_trade["position"] = self.position
        self.current_trade["reward"] = 0.2
        self.current_trade["timesteps"] += 1
        self.current_trade["profit"] = 0
        self.current_trade["parent_ask_bid_lines"] = None
        self.current_trade["balance"] = self.account_balance
        current_image = self.draw_buy_bar(current_image, ask_bid_lines, ask_bid_lines, self.position, self.current_trade["profit"])
        self.trade_queue.append(ask_bid_lines)
        self.image_queue.append(current_image)  
        if self.env_draw:
            self.draw_collection(self.image_queue)            
        self.reward = 0.2 
        
    def hold_position(self,ask_bid_lines, current_image):
        
        self.current_trade["parent_ask_bid_lines"] = self.parent_ask_bid_lines
        self.current_trade["current_price"] = float(f"{self.m1short.iloc[self.current_step]['Bid']}") if self.position == "buy" else float(f"{self.m1short.iloc[self.current_step]['Ask']}")
        self.current_trade["profit"] = self.current_trade["current_price"] - self.current_trade["entry_price"] if self.position == "buy" else self.current_trade["entry_price"] - self.current_trade["current_price"]
        self.current_trade["timesteps"] += 1
        self.current_trade["balance"] = self.account_balance
        current_image = self.draw_buy_bar(current_image, self.current_trade["parent_ask_bid_lines"], ask_bid_lines, self.position , self.current_trade["profit"])
        self.trade_queue.append(ask_bid_lines)
        self.image_queue.append(current_image)
        if self.env_draw:
            self.draw_collection(self.image_queue) 
        if self.current_trade["profit"] >= 0:
            self.reward = 0.6
            self.current_trade["reward"] = self.reward
        else:
            self.reward = -0.6
            self.current_trade["reward"] = self.reward

        
                
    def close_position(self, ask_bid_lines, current_image):
        
        self.current_trade["current_price"] = float(f"{self.m1short.iloc[self.current_step]['Bid']}") if self.position == "buy" else float(f"{self.m1short.iloc[self.current_step]['Ask']}")
        self.current_trade["profit"] = self.current_trade["current_price"] - self.current_trade["entry_price"] if self.position == "buy" else self.current_trade["entry_price"] - self.current_trade["current_price"]
        current_image = self.draw_buy_bar(current_image, self.parent_ask_bid_lines, ask_bid_lines, self.position, self.current_trade["profit"] )
        self.trade_queue.append(ask_bid_lines)
        self.image_queue.append(current_image)
        if self.env_draw:
            self.draw_collection(self.image_queue) 
        if self.current_trade["profit"] >= 0:
            self.reward = 0.9
        else:
            self.reward = -0.9         
        #self.trade_queue, self.image_queue = self.reset_collection(self.trade_queue,self.image_queue)
        self.account_balance += self.current_trade["profit"]
        self.current_trade["current_timestep"] = self.current_step
        self.current_trade["timesteps"] += 1
        self.current_trade["position"] = None
        self.current_trade["balance"] = self.account_balance
        self.current_trade["episode_profit"] = self.episode_profit
        self.position = None
        self.parent_ask_bid = None
        
    def step(self, action):
        self.reward = 0
        self.done = False
        ask_bid_lines, current_image = self.get_ask_bid_lines(f"{self.dataset_directory}{self.m1short.iloc[self.current_step]['image_path']}")
        ask_bid_lines["key"] = self.current_step
        #if self.position == None:
            #self.trade_queue, self.image_queue = self.reset_collection(self.trade_queue,self.image_queue)
        if self.position == None:
            if action == 1:
                self.position = "buy"
                self.prev_trade.append(1)
                self.parent_ask_bid_lines = ask_bid_lines
                self.enter_trade(ask_bid_lines,current_image)
            elif action == 2:
                self.position = "sell"
                self.prev_trade.append(2)
                self.parent_ask_bid_lines = ask_bid_lines
                self.enter_trade(ask_bid_lines,current_image)
            elif action == 0:
                self.position = None
                self.prev_trade.append(0)
                self.parent_ask_bid_lines = ask_bid_lines
                self.enter_idle_mode(ask_bid_lines, current_image)
        elif self.position == "buy":
            if action == 1:
                self.prev_trade.append(1)
                self.hold_position(ask_bid_lines, current_image)
            elif action in [0,2]:
                self.close_position(ask_bid_lines, current_image)
                self.prev_trade.append(0)
                if self.account_balance > int((2/3) * self.chart_decorator.balance_limit) + self.chart_decorator.standard_balance:
                    self.chart_decorator.standard_balance += self.chart_decorator.balance_limit
        elif self.position == "sell":
            if action == 2:
                self.prev_trade.append(2)
                self.hold_position(ask_bid_lines, current_image)
                
            elif action in [1,0]:
                self.close_position(ask_bid_lines, current_image)
                self.position = None
                self.prev_trade.append(0)
                if self.account_balance > int((2/3) * self.chart_decorator.balance_limit) + self.chart_decorator.standard_balance:
                    self.chart_decorator.standard_balance += self.chart_decorator.balance_limit
        self.current_step+=1
        next_state = self.get_obs()
        if self.account_balance < 900:
            self.done = True
        print(self.trade_queue)
        return next_state, self.reward, self.done, self.current_trade
    
    def get_obs(self):
        stacked_obs = np.expand_dims(np.stack([np.asarray(self.image_queue[0])/255, np.asarray(self.image_queue[1])/255, np.asarray(self.image_queue[2])/255, np.asarray(self.image_queue[3])/255]), axis=0)
        
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
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.gamma = 0.99
        self.env = env
        self.actor = self._actor_model()
        self.critic = self._critic_model()
        # self.critic.set_weights(self.actor.get_weights())
        self.models_dir = "models/attention/"
        self.actor_model_name = "att_actor.h5"
        self.critic_model_name = "att_critic.h5"
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
        
    def test_agent(self):
        header = ["step","action", "reward", "done","profit", "Episode Profit", "balance",  "actor", "Eprice", "Cprice", ]
        state, reward, done, current_trade = self.env.reset()
        profits = []
        for step in range(1225):
            rows = []
            action, pred = self.act(state)
            next_state, reward, done, current_trade = self.env.step(action)
            row = [step, action, reward,done, current_trade["profit"], current_trade["episode_profit"], current_trade["balance"], ",".join(str("{:.2f}".format(x)) for x in list(pred)[0][0]), current_trade["entry_price"], current_trade["current_price"]]
            rows.append(row)
            print(tabulate(rows, headers=header, tablefmt='grid'))
            print(f"current_trade {current_trade}\n")
            state = next_state
            profits.append(current_trade["profit"])
            

    def train(self, num_episodes):
        
        header = ["episode","epsilon", "counter","action", "done" ,"profit", "balance","Eprice","Cprice", "critic target_f_values","target_f__reward", "actor pred qvls", "fit actor q_values", "total_profits"]
        self.episode_rewards = []
        target_f_values = []
        for episode in range(num_episodes):
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
                    
            
                self.critic.fit(state, np.array(q_values[0][0][action]).reshape(1,1,1) , verbose=2)
                
                q_values[0][0][action] = target_f
                

                
                self.actor.fit(state, q_values, verbose=2)
                
                state = next_state
                if action == 0 and self.env.prev_trade[2] in [1,2]:
                    self.env.trade_queue, self.env.image_queue = self.env.reset_collection(self.env.trade_queue,self.env.image_queue)
                if counter % 50 == 0:
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
            print(self.episode_rewards)
        return self.episode_rewards
    


    
if __name__ == "__main__":
    action = 3
    state_size = (1, 4, 224, 224, 3)
    chart_decorator = ChartDecorator()
    m1short = pd.read_csv("final_m1short.csv")
    env = ForexCustomEnv(m1short, chart_decorator)
    agent = Agent(state_size, action, env)
    agent.epsilon = 0.15#1.0
    agent.env.env_draw = False
    #agent.load()
    episode_rewards = agent.train(num_episodes=100)
    # agent.test_agent() 
    
