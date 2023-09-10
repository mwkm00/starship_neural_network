import pygame
from nnet import Nnet
import random
import numpy

pygame.init()

WIDTH = HEIGHT = 800
FPS = 30
FONT_SIZE = 12
GENERATION_SIZE = 1000
STARSHIP_SPRITE = "starship\starship.png"

NNET_INPUT_NODES = 6
NNET_HIDDEN_NODES = 12#8
NNET_OUTPUTS = 3#2

font = pygame.font.SysFont("comicsansms", FONT_SIZE)

clock = pygame.time.Clock()
display = pygame.display.set_mode((WIDTH,HEIGHT))

running = True

starship_raw = pygame.image.load(STARSHIP_SPRITE)
starship_raw = pygame.transform.scale(starship_raw, (20,30))

starting_pos = (WIDTH/2, HEIGHT/2)

vec2 = pygame.math.Vector2


class Starship:
    def __init__(self, pos):
        self.pos = vec2(pos)
        self.sprite = starship_raw
        self.rect = self.sprite.get_rect(center=self.pos)
        self.angle = 0
        self.sprite_angle = 0
        self.global_angle = 0
        self.vel = vec2(0,0)
        self.end_vel = vec2(0,-6)
        self.fitness = 0
        self.last_dist = 0
        self.delta_dist = 0
        self.nnet = Nnet(NNET_INPUT_NODES, NNET_HIDDEN_NODES, NNET_OUTPUTS)
    def draw(self):
        rotated_image = pygame.transform.rotate(self.sprite, -self.sprite_angle)
        rotated_image_rect = rotated_image.get_rect(center=self.rect.center).topleft ##BLACKBOX>>
        display.blit(rotated_image, rotated_image_rect)
    def reset(self):
        self.fitness = 0
        self.last_dist = 0
        self.delta_dist = 0

    def rotate_right(self):
        self.sprite_angle += self.vel.length()
        self.angle += self.vel.length()
        self.global_angle += self.vel.length()

    def rotate_left(self):
        self.sprite_angle -= self.vel.length()
        self.angle -= self.vel.length()
        self.global_angle -= self.vel.length()

    def accelerate(self):
        self.vel = self.end_vel
        self.vel.rotate_ip(self.angle)
        self.angle = 0

    def move(self):
        inputs = self.get_inputs()
        outputs = self.nnet.get_outputs(inputs)
        if outputs[0] > 0.5:
            self.rotate_right()
        elif outputs[2] > 0.5:
            self.rotate_left()
            
        if outputs[1] > 0.5:
            self.accelerate()

    def update(self):
        if self.global_angle < 0:
            self.global_angle = 359
        if self.global_angle > 360:
            self.global_angle = 0

        dist = numpy.sqrt(pow(abs(target_coords[0]-self.pos[0]),2) + pow(abs(target_coords[1]-self.pos[1]),2))
        if self.last_dist == 0:
            self.last_dist = dist
        self.fitness += 100/ numpy.sqrt(dist)

        if self.pos[0] + self.vel[0] > -1000 and self.pos[0] + self.vel[0] < 1000:
            if self.pos[1] + self.vel[1] > -1000 and self.pos[1] + self.vel[1] < 1000:
                self.pos += self.vel
        self.rect.center = self.pos
        if self.vel != pygame.math.Vector2(0,0):
            self.vel = self.vel.lerp(pygame.math.Vector2(0,0), 0.1)
        
        self.delta_dist = self.last_dist-dist
        self.last_dist = dist
        
        self.draw()
        self.move()

    def get_inputs(self):
        dist = (numpy.sqrt(pow(abs(target_coords[0]-self.pos[0]),2) + pow(abs(target_coords[1]-self.pos[1]),2))/1414*0.99)+0.01
        angle = (self.global_angle/360*0.99)+0.01
        speed = (self.vel.length()/6*0.99)+0.01
        delta_dist = (self.delta_dist/8*0.99)+0.01
        x_change = (abs(target_coords[0]-self.pos[0])/1414)*0.99+0.01
        y_change = (abs(target_coords[1]-self.pos[1])/1414)*0.99+0.01

        inputs = [
            angle,
            speed,
            dist,
            delta_dist,
            x_change,
            y_change
        ]
        return inputs

starship = Starship((WIDTH/2, HEIGHT/2))
starship.nnet.load_weights()
gametime = 0
generation = 1

while running:
    target_coords = pygame.mouse.get_pos()
    dt = clock.tick(FPS)
    gametime += dt

    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    display.fill(0)
    
    starship.update()
    
    pygame.display.update()
