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
NNET_HIDDEN_NODES = 12
NNET_OUTPUTS = 3

MUTATION_CUT_OFF = 0.4
MUTATION_TOP_CUT_OFF = 0.2
MUTATION_BAD_TO_KEEP = 0.2
MUTATION_MODIFY_CHANCE_LIMIT = 0.2

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
        #self.vel = vec2(0,0)
        #self.pos = vec2(50,700)
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
        dist = numpy.sqrt(pow(abs(training_coords[0]-self.pos[0]),2) + pow(abs(training_coords[1]-self.pos[1]),2))
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
        dist = (numpy.sqrt(pow(abs(training_coords[0]-self.pos[0]),2) + pow(abs(training_coords[1]-self.pos[1]),2))/1414*0.99)+0.01
        angle = (self.global_angle/360*0.99)+0.01
        speed = (self.vel.length()/6*0.99)+0.01
        delta_dist = (self.delta_dist/8*0.99)+0.01
        x_change = (abs(training_coords[0]-self.pos[0])/1414)*0.99+0.01
        y_change = (abs(training_coords[1]-self.pos[1])/1414)*0.99+0.01

        inputs = [
            angle,
            speed,
            dist,
            delta_dist,
            x_change,
            y_change
        ]
        return inputs
    def create_offspring(p1,p2):
        new_starship = Starship(starting_pos)
        new_starship.nnet.create_mixed_weights(p1.nnet, p2.nnet)
        return new_starship

class StarshipCollection:
    def __init__(self):
        self.starships = []
        self.create_new_generation()
        self.best_starship = Starship(starting_pos)
    def create_new_generation(self):
        self.starships = []
        for x in range(GENERATION_SIZE):
            self.starships.append(Starship(starting_pos))

    def update(self):
        for s in self.starships:
            s.update()
            
    def evolve_pop(self):
        for s in self.starships:
            s.fitness += 10000/numpy.sqrt(pow(abs(training_coords[0]-s.pos[0]),2) + pow(abs(training_coords[1]-s.pos[1]),2))

        self.starships.sort(key=lambda x: x.fitness, reverse=True)
        self.best_starship = self.starships[0]

        cut_off = int(len(self.starships) * MUTATION_CUT_OFF)
        top_cut_off = int(len(self.starships) * MUTATION_TOP_CUT_OFF)
        good_starships = self.starships[0:cut_off]
        top_starships = self.starships[0:top_cut_off]
        bad_starships = self.starships[cut_off:]
        num_bad_to_take = int(len(bad_starships)* MUTATION_BAD_TO_KEEP)

        for s in bad_starships:
            s.nnet.modify_weights()

        new_starships = []

        idx_bad_to_take = numpy.random.choice(numpy.arange(len(bad_starships)), num_bad_to_take, replace=False)

        for i in idx_bad_to_take:
            new_starships.append(bad_starships[i])
        
        new_starships.extend(good_starships)
        children_needed = len(self.starships) - len(new_starships)
        for t in range(children_needed):
            idx_to_breed = numpy.random.choice(numpy.arange(len(top_starships)), 2, replace=False)
            if idx_to_breed[0] != idx_to_breed[1]:
                new_starship = Starship.create_offspring(top_starships[idx_to_breed[0]], top_starships[idx_to_breed[1]])
                if random.random() < MUTATION_MODIFY_CHANCE_LIMIT:
                    new_starship.nnet.modify_weights()
                new_starships.append(new_starship)
        
        for s in new_starships:
            s.reset()

        self.starships = new_starships
        

starships = StarshipCollection()

gametime = 0
training_coords = [700, 100]
generation = 1

def display_info():
    time_elapsed_text = font.render(f"Time elapsed: {gametime/1000}", True, (255,255,255))
    display.blit(time_elapsed_text, (0,0))
    text_generation = font.render(f"Generation: {generation}", True, (255,255,255))
    display.blit(text_generation, (0,20))
    gen_size = font.render(f"Generation size: {len(starships.starships)}", True, (255,255,255))
    display.blit(gen_size, (0,40))

while running:
    dt = clock.tick(FPS)
    gametime += dt

    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                starships.best_starship.nnet.save_weights()
                print("WEIGHTS SUCCESSFULLY SAVED")
            
    display.fill(0)

    if gametime/1000 > 20:
        training_coords = (random.randint(0, 800), random.randint(0,800))
        gametime = 0
        generation += 1
        starships.evolve_pop()
    
    pygame.draw.circle(display, (0,255,0), training_coords, 10)    
    starships.update()
    
    display_info()
    pygame.display.update()
