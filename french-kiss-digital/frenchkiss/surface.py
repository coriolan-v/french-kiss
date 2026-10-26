"""A smooth, bone-weighted visual surface over the rigid collision backbone."""
import math
from .config import CONFIG


def tongue_profile(u):
    """Broad, nearly parallel sides with a rounded human-like end."""
    width=CONFIG['tongue_half_width_m']*(.92+.08*math.sin(math.pi*u))
    thickness=CONFIG['tongue_half_thickness_m']*(1-.18*u)
    if u>.85:
        rounding=math.sqrt(max(.0004,1-((u-.85)/.15)**2))
        width*=rounding
        thickness*=rounding
    return width,thickness


def tongue_skin(side,sign,length,segments,material):
    anchor=CONFIG['anchor_half_span_m']
    rings=49; around=24
    vertices=[]; faces=[]; weights=[[] for _ in range(segments)]
    ids=[[] for _ in range(segments)]
    for r in range(rings):
        u=r/(rings-1)
        x=length*u
        width,thickness=tongue_profile(u)
        for k in range(around):
            angle=k*2*math.pi/around
            y=width*math.copysign(abs(math.cos(angle))**.65,math.cos(angle))
            # A shallow central groove gives a tongue-like upper surface.
            z=thickness*math.copysign(abs(math.sin(angle))**.65,math.sin(angle))
            if z>0: z-=.004*math.exp(-(y/.025)**2)*math.sin(math.pi*u)
            vertices.extend([sign*anchor-sign*x,-sign*y,1.25+z])
            bone=max(0,min(segments-1,x/(length/segments)-.5))
            first=int(bone); second=min(segments-1,first+1)
            fraction=bone-first
            index=r*around+k
            ids[first].append(index); weights[first].append(1-fraction)
            if second!=first and fraction>0:
                ids[second].append(index); weights[second].append(fraction)
            if r<rings-1:
                a=index; b=r*around+(k+1)%around; c=b+around; d=a+around
                # Outward winding for world orientation on each side.
                faces.extend([a,c,b,a,d,c])
    for ring,bone,local_x in [(0,0,0), (rings-1,segments-1,length)]:
        center=len(vertices)//3
        vertices.extend([sign*anchor-sign*local_x,0,1.25])
        ids[bone].append(center); weights[bone].append(1.)
        for k in range(around):
            a=ring*around+k; b=ring*around+(k+1)%around
            faces.extend([center,a,b] if ring==0 else [center,b,a])
    vertex=' '.join(f'{v:.6f}' for v in vertices)
    face=' '.join(map(str,faces))
    result=[f'<skin name="{side}_surface" material="{material}" vertex="{vertex}" face="{face}" inflate=".002">']
    for i in range(segments):
        bindx=sign*anchor-sign*i*length/segments
        quat='1 0 0 0' if sign==-1 else '0 0 0 1'
        result.append(f'<bone body="{side}_link{i}" bindpos="{bindx} 0 1.25" bindquat="{quat}" vertid="{" ".join(map(str,ids[i]))}" vertweight="{" ".join(map(str,weights[i]))}"/>')
    result.append('</skin>')
    return '\n'.join(result)
