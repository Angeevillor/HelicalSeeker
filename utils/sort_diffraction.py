from PIL import Image,ImageDraw
import re
import cv2


def read_view_region(values_file):
    """Read Diffraction_simulation's y_min,y_max,x_min,x_max metadata."""
    try:
        with open(values_file, encoding="utf-8") as f:
            for line in f:
                if line.strip().lower().startswith("region"):
                    payload = line.split(":", 1)[1] if ":" in line else line
                    values = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?", payload)]
                    if len(values) >= 4:
                        return values[:4]
    except (OSError, ValueError):
        pass
    return None
def extract_number(filename):
    match = re.search(r"\d+", filename)
    if match:
        return int(match.group())
    else:
        return 0

def converge(img1,img2):
    w1,h1=img1.size
    # cv2 expects (width, height); keep the reference and simulation image
    # coordinate systems aligned even when an old non-square image is used.
    img2=cv2.resize(img2,(w1,h1))

    img2=Image.fromarray(img2)

    w2,h2=img2.size

    W=(w1+w2)/2
    H=max(h1,h2)
    new_img=Image.new("RGB",(int(W),int(H)))
    left=img1.crop((0,0,w1//2,h1))
    right=img2.crop((w2//2,0,w2,h2))

    new_img.paste(left,(0,0))
    new_img.paste(right,(int(W//2),0))

    return new_img
def convert(img,point,X,PZ):
    dic={}
    a,b=img.size
    if 0<= point[0]<=a//2 and 0<=point[1]<=b//2:
        dic["coordinate"]=point
        dic["layer order"]=1
    elif 0 <= point[0] <= a and point[1] <0:
        count=1
        while point[1]<0:
            point[1]+=PZ
            point[0]-=X
            count+=1
            if (point[0]<=0 or point[1]>=b//2):
                break
        if 0 <= point[0] <= a // 2 and 0 <= point[1] < b // 2:
            dic["coordinate"] = point
            dic["layer order"] =count
        else:
            dic["coordinate"] = []
            dic["layer order"] = -1
    else:
        dic["coordinate"] = []
        dic["layer order"] = -1
    return dic
def generate_lattice(region,X,PZ):
    dic={}
    a,b=region,region
    center_x,center_y=a//2,b//2
    X_step=round(center_x/abs(X))
    Y_step=round(center_y/abs(PZ))
    step=min(X_step,Y_step)
    if step>0:
        for i in range(-step,step+1):
            x=center_x-i*X
            y=center_y-i*PZ
            dic["coordinate_order({})".format(i)]=[x,y]
            dic["step"]=step
    else:
        dic["coordinate_order({})".format(0)]=[center_x,center_y]
        dic["step"]=0
    return dic
def shift_lattice(img,region,X,PZ,Z,n,view_region=None):
    """Generate lattice points in the coordinates of the output image.

    ``region`` is the side length used when the simulated diffraction
    pattern was generated (for example 2048), while ``img`` may now be the
    512x512 square screenshot produced by Diffraction_simulation.  Generate
    the lattice in the original coordinate system first, then map offsets
    around its centre into the actual image coordinates.  This keeps the
    physical lattice spacing correct after screenshot downsampling.
    """
    dic={}
    region=float(region)
    if region <= 0:
        raise ValueError("Diffraction region must be positive")
    dic_1=generate_lattice(region,X,PZ)
    a,b=img.size
    center_x,center_y=a/2,b/2
    source_center=region/2.0
    if view_region is not None and len(view_region) >= 4:
        # Metadata order is y_min,y_max,x_min,x_max.  This maps the original
        # FFT coordinates into the 512-pixel screenshot, including zoomed
        # central crops.
        view_y_min, view_y_max, view_x_min, view_x_max = map(float, view_region[:4])
        # Older Diffraction_simulation versions wrote the y limits reversed;
        # normalize both generations before calculating the scale.
        view_y_min, view_y_max = sorted((view_y_min, view_y_max))
        view_x_min, view_x_max = sorted((view_x_min, view_x_max))
        view_width = view_x_max - view_x_min
        view_height = view_y_max - view_y_min
        if view_width > 0 and view_height > 0:
            scale_x = a/view_width
            scale_y = b/view_height
            map_x = lambda source_x: (source_x-view_x_min)*scale_x
            map_y = lambda source_y: (source_y-view_y_min)*scale_y
        else:
            scale_x = a/region
            scale_y = b/region
            map_x = lambda source_x: center_x+(source_x-source_center)*scale_x
            map_y = lambda source_y: center_y+(source_y-source_center)*scale_y
    else:
        scale_x = a/region
        scale_y = b/region
        map_x = lambda source_x: center_x+(source_x-source_center)*scale_x
        map_y = lambda source_y: center_y+(source_y-source_center)*scale_y
    step=dic_1["step"]
    if step>0:
        for j in range(-n,n+1):
            for i in range(-step,step+1):
                # Work in the original simulation coordinate system, then
                # map the displacement from the meridian to the screenshot.
                source_x=source_center-i*X
                source_y=source_center-i*PZ+j*Z
                x=map_x(source_x)
                y=map_y(source_y)
                if 0<=x<=a and 0<=y<=b:
                    dic["coordinate_order({})_pattern({})".format(i,-j)]=[x,y]
                    dic["step"]=step
                    dic["shift"]=n
                else:
                    pass
    else:
        dic["coordinate_order({})_pattern(0)".format(0)]=[center_x,center_y]
        dic["step"]=0
    return dic


