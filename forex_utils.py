from PIL import Image
import numpy as np
import pandas as pd

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
        if position == -1:
            x1, y1 = 160, 204
            x2, y2 = 175, 224
            draw_context.rectangle([x1, y1, x2, y2], fill=(255, 255, 0)) 
        elif position == 0:
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
            if position == -1 and profit > 0 and self.prev_trade[2] == 0:
                x1, y1 = 200, 0
                x2, y2 = 224, 14
                draw_context.rectangle([x1, y1, x2, y2], fill=(255, 255, 0))
            elif position == -1 and profit > 0 and self.prev_trade[2] == 1:
                x1, y1 = 200, 0
                x2, y2 = 224, 14
                draw_context.rectangle([x1, y1, x2, y2], fill=(0, 255, 0))
            elif position == -1 and profit < 0 and self.prev_trade[2] in [1,0]:
                x1, y1 = 200, 0
                x2, y2 = 224, 14
                draw_context.rectangle([x1, y1, x2, y2], fill=(255,0, 0))
       
    def draw_buy_bar(self, img2, start_lines, end_lines, position = None, profit = 0, prev_trade=None, account_balance = 1100 ):
        self.prev_trade = prev_trade
        print(f"\n\n\nDraw_buy_bar {position}\n\n\n")
        from PIL import Image, ImageDraw
        d = ImageDraw.Draw(img2)
        
        self.add_top_bottom_bar(img2, d)
        self.draw_account_balance(img2, d, account_balance)

        if position == 1 and 1 in [self.prev_trade[3]]:
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
        elif position == 0 and 0 in [self.prev_trade[3]]:
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


        elif position in [-1] and self.prev_trade[3] == -1:
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

            if self.prev_trade[2] == 1 and profit > 0:   
                #print right bar for the trade that has closed
                line_color = (0, 255, 0)
                top = (213, start_lines["ask_line"])
                bottom = (213, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10) 

                line_color = (241, 254, 198)
                top = (190, start_lines["ask_line"])
                bottom = (190, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10)             

            elif self.prev_trade[2] == 1 and profit < 0:
                line_color = (0, 255, 0)
                top = (213, start_lines["ask_line"])
                bottom = (213, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10) 

                line_color = (255,0,0)
                top = (190, start_lines["ask_line"])
                bottom = (190, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10)    
            elif self.prev_trade[2] == 0 and profit > 0:
                # print(f"\n\nHey I was here\n\n{self.prev_trade}")
                
                line_color = (255, 255, 0)
                top = (213, start_lines["ask_line"])
                bottom = (213, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10) 

                line_color = (241, 254, 198)
                top = (190, start_lines["ask_line"])
                bottom = (190, end_lines["bid_line"])
                d.line([top, bottom], fill=line_color, width=10)    

            elif self.prev_trade[2] == 0 and profit < 0:
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
