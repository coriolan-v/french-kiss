"""Calibrated virtual stereo marker tracking, with noise and dropout.

Synthetic marker projections use perfect-state sites as camera input; this is
not image detection, and learned agents still use perfect-state observations.
"""
import numpy as np
import cv2

class StereoTracker:
    def __init__(self,seed=0):
        self.rng=np.random.default_rng(seed)
        self.K=np.array([[700.,0,480],[0,700,300],[0,0,1]])
        self.projections=[]
        for eye in (np.array([-.6,-3.,2.]),np.array([.6,-3.,2.])):
            forward=np.array([0,0,1.1])-eye
            forward/=np.linalg.norm(forward)
            right=np.cross(forward,[0,0,1]); right/=np.linalg.norm(right)
            down=np.cross(forward,right)
            rotation=np.stack([right,down,forward])
            self.projections.append(self.K@np.column_stack([rotation,-rotation@eye]))

    def observe(self,points,noise_px=.6,dropout=.05):
        points=np.asarray(points,dtype=float)
        hom=np.column_stack([points,np.ones(len(points))])
        pixels=[]
        visible=np.ones(len(points),dtype=bool)
        for projection in self.projections:
            q=hom@projection.T
            uv=q[:,:2]/q[:,2:]
            uv+=self.rng.normal(0,noise_px,uv.shape)
            visible &= (q[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<600)
            visible &= self.rng.random(len(points))>=dropout
            pixels.append(uv)
        reconstructed=cv2.triangulatePoints(*self.projections,pixels[0].T,pixels[1].T)
        reconstructed=(reconstructed[:3]/reconstructed[3]).T
        reconstructed[~visible]=np.nan
        return {'pixels':pixels,'visible':visible,'positions':reconstructed}
