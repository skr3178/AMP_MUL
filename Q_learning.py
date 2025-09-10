import pygame   #Importe la bibliothèque pygame pour créer le jeu et l'initialiser
import numpy as np  #Importe numpy en tant que np
import random
from paddle import Paddle   #Importe la class Paddle dans paddle.py
from balle import Balle   #Importe la class Balle dans balle.py
from mur import Mur   #Importe la class Mur dans mur.py
from simple_ai import SimpleAI  #Importe la class SimpleAI dans simple_ai.py

#Définir les couleurs à utiliser
BLANC = (255,255,255)
NOIR = (0, 0, 0)
BLEUFONCE = (36 ,90 ,140)
BLEU = (0, 190, 242)
ROUGE = (204, 53, 53)
VERT = (80, 162, 45)
VIOLET = (156, 60, 185)
ORANGE = (255, 150, 31)
JAUNE = (245, 210, 10)

#Dimensions écran
screen_width = 800
screen_height = 600

#Actions possibles par l'IA
actions = ['GAUCHE', 'DROITE', 'RIEN']

# Q-learning avec états simplifiés
#Discrétiser pour l'IA
nb_pos_paddle = 10
nb_pos_x_balle = 12
nb_pos_y_balle = 12
nb_vx = 2   # gauche/0/droite
nb_vy = 2   # haut/0/bas

# Q-learning avec états simplifiés
Q = np.zeros((nb_pos_paddle,
              nb_pos_x_balle,
              nb_pos_y_balle,
              nb_vx, 
              nb_vy,
              len(actions)))

alpha = 0.2
gamma = 0.9

#Reduire la taille de l'écran pour moins de possibilités
def get_state(balle, paddle, screen_width, screen_height):
    x = int(balle.rect.x / (screen_width / nb_pos_x_balle))
    y = int(balle.rect.y / (screen_height / nb_pos_y_balle))
    vx = 0 if balle.velocity[0] < 0 else 1
    vy = 0 if balle.velocity[1] < 0 else 1
    r = int(paddle.rect.x / (screen_width / nb_pos_paddle))

    #Eviter les dépassements
    x = min(nb_pos_x_balle - 1, x)
    y = min(nb_pos_y_balle - 1, y)
    r = min(nb_pos_paddle - 1, r)

    return (r, x, y, vx, vy)

# Choisir une action
def choose_action(state, epsilon):
    if random.uniform(0,1) < epsilon:
        return random.choice(actions)
    else:
        action_index = np.argmax(Q[state + (slice(None),)])
        return actions[action_index]

# Mise à jour Q-learning
def update_Q(state, action, reward, next_state):
    action_index = actions.index(action)
    old_value = Q[state + (action_index,)]
    next_max = np.max(Q[next_state + (slice(None),)])
    Q[state + (action_index,)] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)






#Fonction principale
def run_game(total_games):

    vies = 1
    score = 0
    game_speed = 10000000  # vitesse par défaut
    
    enCours  = True
    pygame.init()   #Initialise le jeu pygame

    #Initialiser le paddle et ses caractéristiques
    paddle = Paddle(BLANC, 100, 20)
    paddle.rect.x = 350
    paddle.rect.y = 560

    ##Initialiser la balle et ses caractéristiques
    balle = Balle(BLANC, 15, 15)
    balle.rect.x = 345
    balle.rect.y = 300

    #Créer une liste qui contiendra tout les sprites du jeu
    sprites_liste = pygame.sprite.Group()

    #Ajouter les sprites seuls à la liste des sprites
    sprites_liste.add(paddle)
    sprites_liste.add(balle)

    #Créer 3 lignes de briques et les ajouter au groupe mur_briques pour créer un mur
    mur_briques = pygame.sprite.Group()
    for i in range(10):
        brique = Mur(VIOLET,80,30)
        brique.rect.x = i* 80
        brique.rect.y = 70
        sprites_liste.add(brique)
        mur_briques.add(brique)
    for i in range(10):
        brique = Mur(ROUGE,80,30)
        brique.rect.x = i* 80
        brique.rect.y = 100
        sprites_liste.add(brique)
        mur_briques.add(brique)
    for i in range(10):
        brique = Mur(ORANGE,80,30)
        brique.rect.x = i* 80
        brique.rect.y = 130
        sprites_liste.add(brique)
        mur_briques.add(brique)
    for i in range(10):
        brique = Mur(JAUNE,80,30)
        brique.rect.x = i* 80
        brique.rect.y = 160
        sprites_liste.add(brique)
        mur_briques.add(brique)


    #Réglages de la fenêtre
    tailleEcran = (800, 600)
    ecran = pygame.display.set_mode(tailleEcran)
    pygame.display.set_caption("BreakIA")

    #L'horloge pour contrôler la fréquence de rafraichissement
    horloge = pygame.time.Clock()

    #Boucle principale
    while enCours:
        #Variables de vies, score, enCours et vitesse du jeu pour la boucle princiaple


        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None  # signal pour arrêter complètement le programme
            
            #Mode basse vitesse (à corriger)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    # Toggle entre lent et rapide
                    if game_speed == 10000000:
                        game_speed = 60   # mode très lent
                    else:
                        game_speed = 10000000   # revenir à normal

            #Mode pause quand on appuye sur ECHAP
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                while True: #Boucle infinie tant qu'on appuye pas sur ESC
                    event = pygame.event.wait()
                    police = pygame.font.Font("C:/Users/alexw/Desktop/Code/Python/Projet_BreakAI/ressources/font/dogica.otf", 64)
                    texte = police.render("PAUSE", 1, BLANC)
                    ecran.blit(texte, (248,300))
                    pygame.display.flip()
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return None  # signal pour arrêter complètement le programme
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        break #Retourner au jeu
                    
        
        #Updater les différents sprites chaque frame
        sprites_liste.update()

        if balle.rect.x >= 790:
            balle.velocity[0] = -balle.velocity[0]
        if balle.rect.x <= 10:
            balle.velocity[0] = -balle.velocity[0]
        if balle.rect.y <= 40:
            balle.velocity[1] = -balle.velocity[1]
        if balle.rect.y >= 595: #Mur du bas
            balle.velocity[1] = -balle.velocity[1]
            vies -= 1   #Quand la balle tombe en dessous du paddle, on perd une vie
            #S'il n'y a plus de vies ou si on a cassé toutes les briques on relance
            if vies <= 0 or len(mur_briques) == 0:
                return score  #On sort proprement de run_game et on renvoie le score

        
        #Gérer la collision balle paddle, et faire rebondir la balle
        if pygame.sprite.collide_mask(balle, paddle):
            balle.rect.x -= balle.velocity[0]
            balle.rect.y -= balle.velocity[1]
            balle.rebond()

        #Gérer la collision balle brique, et faire rebondir et disparaître une brique
        collision_briques = pygame.sprite.spritecollide(balle, mur_briques, False) #Liste de tout les sprites qui entrent en collision avec la balle
        for brique in collision_briques:
            balle.rect.x -= balle.velocity[0]
            balle.rect.y -= balle.velocity[1]
            balle.rebond()
            score += 1  #Augmenter le score quand une brique est touchée
            brique.kill()


        #Eviter que la balle reste coincée à l'horizontale et à la verticale
        if balle.velocity[1] == 0:
            balle.velocity[1] += 2
        if balle.velocity[0] == 0:
            balle.velocity[0] += 2


        #Code d'affichage des éléments
        #Mettre une couleur à l'écran
        ecran.fill(NOIR)
        pygame.draw.line(ecran, BLANC, [0, 40], [800, 40], 2)

        #Afficher le score et les vies en haut de l'écran
        police = pygame.font.Font("C:/Users/alexw/Desktop/Code/Python/Projet_BreakAI/ressources/font/dogica.otf", 32)  #Choisir une police
        texte = police.render("Score: " + str(score), 0, BLANC)
        ecran.blit(texte, (25,9))
        texte = police.render("Vies: " + str(vies), 0, BLANC)
        ecran.blit(texte, (550,9))

        #Afficher tout les sprites à l'écran avec cette ligne
        sprites_liste.draw(ecran)
    
        #Afficher à l'écran ce qu'on vient de définir en haut
        pygame.display.flip()
        

        # État courant
        state = get_state(balle, paddle, screen_width, screen_height)
        # epsilon décroissant pour plus d’exploitation
        epsilon = max(0.05, 1 - (total_games / 1000))
        action = choose_action(state, epsilon)

        if action == 'DROITE':
            direction = 'DROITE'
        elif action == 'RIEN':
            direction = 'RIEN'
        elif action == 'GAUCHE':
            direction = 'GAUCHE'

                # Déplacement
        if direction == 'DROITE':
            paddle.rect.x += 7
        elif direction == 'GAUCHE':
            paddle.rect.x -= 7
        elif direction == 'RIEN':
            paddle.rect.x += 0

        if paddle.rect.x < 0:
            paddle.rect.x = 0
        if paddle.rect.x > screen_width - paddle.rect.width:
            paddle.rect.x = screen_width - paddle.rect.width


        # Récompense
        reward = -0.01
        if collision_briques:
            reward = 1
        # Si la balle tombe en bas de l’écran
        if balle.rect.y > screen_height:
            reward = -5
        if len(mur_briques) == 0:
            reward = 10



        next_state = get_state(balle, paddle, screen_width, screen_height)
        update_Q(state, action, reward, next_state)

        with open("C:/Users/alexw/Desktop/Code/Python/Projet_BreakAI/scores/scores.log", "a") as f:
            f.write(f"{total_games},{score}\n")


        #Limite à 60 fps
        pygame.display.update()
        horloge.tick(game_speed)
    pygame.quit


def main():
    total_games = 0
    while True:
        score = run_game(total_games)
        if score is None:   # <- si la fenêtre a été fermée
            break
        print(f"Partie {total_games} terminée - Score: {score}")
        total_games += 1
    pygame.quit()

#Lancer le jeu seulement si le fichier est exécuté directement
if __name__ == "__main__":
    main()