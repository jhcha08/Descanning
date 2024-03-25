import cv2
import numpy as np
import random


def uint2single(img):
    return np.float32(img/255.)

def degrade_chroma(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = s.astype(np.float32)
    s *= factor
    s = np.clip(s, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_gaussian_noise(image):
    mean, sigma = np.random.randint(5,10), np.random.randint(10,20)
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gaussian, 0, 255).astype(np.uint8)
   
    return noisy_image

def apply_blending(img_clean, img_blend):
    if random.uniform(0, 1) <= 0.7:
        alpha = random.uniform(0.85, 0.95)
        img_clean = cv2.resize(img_clean, (img_blend.shape[1], img_blend.shape[0]))
        return cv2.addWeighted(img_clean, alpha, img_blend, 1-alpha, 0)
    
    return img_clean

def apply_chroma_degradation(img):
    threshold = random.uniform(0, 1)
    intensity = random.uniform(0.2, 0.4) if threshold <= 0.8 else random.uniform(1.2, 1.5)
    
    return degrade_chroma(img, intensity)

def apply_linear_laser_patterns(img):
    h, w = img.shape[:2]
    if random.uniform(0, 1) <= 0.5:
        for _ in range(random.randint(1, 5)):
            y = random.randint(0, h)
            x = random.randint(0, w)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.line(img, (0, y), (w, y), color, 1)
            cv2.line(img, (x, 0), (x, h), color, 1)

    return img

def apply_dust_distortion(img):
    h, w = img.shape[:2]
    if random.uniform(0, 1) <= 0.3:
        mask_dust = np.zeros(img.shape[:2], dtype=np.uint8)
        for _ in range(200):
            center, radius = (random.randint(0, h), random.randint(0, w)), random.randint(1, 5)
            cv2.circle(mask_dust, center, radius, 255, -1)
        img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask_dust))

    return img

def transform_image(img_clean, img_blend):
    img_degraded = apply_blending(img_clean, img_blend)
    img_degraded = apply_chroma_degradation(img_degraded)
    img_degraded = apply_gaussian_noise(img_degraded)
    img_degraded = apply_linear_laser_patterns(img_degraded)
    img_degraded = apply_dust_distortion(img_degraded)
    img_scan = img_degraded

    return img_clean, img_scan
