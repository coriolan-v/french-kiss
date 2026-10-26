"""Mouse controls for the digital joint test panel."""
import numpy as np
import pygame
from frenchkiss.model import JOINT_NAMES


class JointPanel:
    def __init__(self,test):
        self.test=test
        self.buttons=[]
        self.sliders=[]

    def event(self,event):
        if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
            for rect,action in self.buttons:
                if rect.collidepoint(event.pos): action(); return
        if event.type==pygame.MOUSEBUTTONDOWN or (event.type==pygame.MOUSEMOTION and event.buttons[0]):
            if event.type==pygame.MOUSEBUTTONDOWN and event.button!=1: return
            for index,rect in self.sliders:
                if rect.inflate(0,18).collidepoint(event.pos):
                    self.test.select(index)
                    low,high=self.test.limits[index]
                    self.test.set_target(low+(high-low)*np.clip((event.pos[0]-rect.x)/rect.w,0,1))
                    return

    def draw(self,screen,text,small,title):
        t=self.test
        self.buttons=[]; self.sliders=[]
        color=(239,176,104)
        text('JOINT TEST',990,25,color,title)
        text('Tendon physics' if t.physical else 'Pose preview / physics paused',990,67,face=small)
        def button(label,rect,action,active=False):
            rect=pygame.Rect(rect)
            pygame.draw.rect(screen,(95,66,44) if active else (40,47,63),rect,border_radius=6)
            text(label,rect.x+8,rect.y+6,face=small)
            self.buttons.append((rect,action))
        for n,s in enumerate(('A','B')):
            button('Tongue '+s,(992+n*128,103,120,33),lambda n=n:t.select(n*4+t.selected%4),t.selected//4==n)
        for joint,name in enumerate(JOINT_NAMES):
            index=(t.selected//4)*4+joint
            y=157+joint*100
            button(name.replace('_',' ').title(),(992,y,250,30),lambda index=index:t.select(index),index==t.selected)
            rect=pygame.Rect(1000,y+43,235,9)
            low,high=t.limits[index]
            value=t.targets[index]
            pygame.draw.rect(screen,(48,54,69),rect,border_radius=4)
            x=int(rect.x+(value-low)/(high-low)*rect.w)
            pygame.draw.circle(screen,color,(x,rect.centery),6)
            self.sliders.append((index,rect))
            actual=t.sim.data.qpos[index]
            text(f'Target {np.degrees(value):+.1f} / actual {np.degrees(actual):+.1f} deg',995,y+62,face=small)
        text(f'Range {np.degrees(t.limits[t.selected,0]):.1f} to {np.degrees(t.limits[t.selected,1]):.1f} deg',993,568,face=small)
        button('Sweep joint',(992,600,120,34),lambda:t.start_sweep(),t.sweep and not t.sweep_all)
        button('Test all 8',(1120,600,122,34),lambda:t.start_sweep(True),t.sweep and t.sweep_all)
        button('Minimum',(992,645,80,34),lambda:t.set_target(t.limits[t.selected,0]))
        button('Zero',(1077,645,80,34),lambda:t.set_target(0))
        button('Maximum',(1162,645,80,34),lambda:t.set_target(t.limits[t.selected,1]))
        button('Center all',(992,690,120,34),t.center)
        def toggle():
            t.physical=not t.physical
            t.sim.data.qvel[:]=0
        button('Physics on/off',(1120,690,122,34),toggle,t.physical)
        text(f'Commands swept {len(set(t.completed))}/8' if t.sweep_all else ('Sweeping '+t.name if t.sweep else 'Manual / agents disabled'),993,736,face=small)
        text('J  return to interaction',993,767,face=small)
