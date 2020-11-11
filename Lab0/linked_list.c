/*** ALGO-PROG - Ensimag 1A apprentis
     Karine Altisen
***/
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>


struct _cell {
    int val;
    struct _cell * next;
};


typedef struct _cell * list;


list create_cell (int v) {
    list res;
    res = malloc(sizeof(struct _cell));
    res->val = v;
    res->next = NULL;
    return res;
}


/* affichage de la liste dans l'ordre de parcours */
void print (list l) {
    list cur = l;
    printf("[");
    while (cur != NULL) {
        printf("%d, ", cur->val);
        cur = cur->next;
    }
    printf("]");
}

/* recherche sequentielle d'un element dans la liste */
bool search (int v, list l) {
    list cur = l;
    while (cur != NULL) {
        if (cur->val == v) return true;
        cur = cur->next;
    }
    return false;
}

/* insertion de l'element en tete de liste */
void add_first (int v, list * l) {
    list old = *l;
    *l = create_cell(v);
    (*l)->next = old;
}

/* insertion de l'element en queue de liste : solution avec pointeur courant */
void add_last_1 (int v, list * l) {

    if (*l == NULL) {
        *l = create_cell(v);
    } else {
        list cur = *l;
        while (cur->next != NULL) {
            cur = cur->next;
        }
        cur->next = create_cell(v);
    }
}

/* insertion de l'element en queue de liste : solution avec pointeur courant 
   et pointeur precedent */
void add_last_2 (int v, list * l) {

    if (*l == NULL) {
        *l = create_cell(v);
    } else {
        list prev = *l;
        list cur = prev->next;
        while (cur != NULL) {
            prev = cur;
            cur = cur->next;
        }
        prev->next = create_cell(v);
    }
}


/* suppression du premier element de la liste */
int remove_first (list * l) {
    // precond: *l != NULL
    assert (*l != NULL);

    list tmp = *l;
    int res = (*l)->val;
    *l = (*l)->next;
    free(tmp);
    return res;
}

/* suppression du dernier element de la liste :
   solution avec un pointeur courant et un pointeur précédent */
int remove_last_1(list * l) {
    // précondition : *l != NULL
    assert (*l != NULL);
    
    // ici le cas particulier est quand l n'a qu'un élément
    if ((*l)->next == NULL) {
        int v = (*l)->val;
        free(l);
        *l = NULL;
        return v;
    } else {
        list prev; list cur;
        prev = *l; 
        cur = (*l)->next;
        while (cur->next != NULL) {
            prev = cur;
            cur = cur->next;
         }
         prev->next = NULL;
         int v = cur->val;
         free(cur);
         return v;
    }
}

/* suppression du dernier element de la liste :
   solution avec un pointeur courant */
int remove_last_2 (list * l) {
    // précondition : *l != NULL
    assert (*l != NULL);

    // ici le cas particulier est quand l n'a qu'un élément
    if ((*l)->next == NULL) {
        int v = (*l)->val;
        free(l);
        *l = NULL;
        return v;
    } else {
        list cur = *l;
        // cur != null et cur->next != null
         while (cur->next->next != NULL) {
            cur = cur->next;
         }
         // cur->next->next == null
         list tmp = cur->next;
         int v = tmp->val;
         cur->next = NULL;
         free(tmp);
         return v;
     }
}


/* suppression du dernier element de la liste :
   l n'a pas de sentinelle, on lui en ajoute une temporairement */
int remove_last_3 (list * l) {
    // précondition : *l != NULL
    assert (*l != NULL);
    
    list prev = create_cell(0); // sentinelle, la valeur est arbitraire
    prev->next = *l;
    *l =  prev;

    list cur = *l;
    // ici cur != null
    while (cur->next != NULL) {
       prev = cur;
       cur = cur->next;
    }
    prev->next = NULL;
    int v = cur->val;
    free(cur);
    // libérer la sentinelle
    cur = *l;
    *l = (*l)->next;
    free(cur);
    return v;
}

/* suppression d'une valeur donnée : solution avec sentinelle */
void my_remove (int v, list * l) {
    list prev =  create_cell(0); // sentinelle, valeur arbitraire
    prev->next = *l;
    *l =  prev;
    list cur = (*l)->next;
    while (cur != NULL) {
         if (cur->val == v) {
            prev->next = cur->next;
            free(cur);
            break;
         }
         prev = prev->next;
         cur = cur->next;
     }
     // libérer la sentinelle
     cur = *l;
     *l = (*l)->next;
     free(cur);
}

/* inversion (sans allocation ! et en temps lineaire) */
void inverse (list * l) {
    list todo, done;
        
    done = NULL;
    todo = *l;
    while (todo != NULL) {
        list aux = todo->next;
        todo->next = done;
        done = todo;
        todo = aux;
    }
    *l = done;
}

/* Solution pour liste vide valant sentinelle : insertion en queue
    avec hypothèse d'une sentinelle en tête de liste
    Dans ce cas la liste à 0 éléments est caractérisée par l->next=null 
*/
void add_last_sent (list * l, int v) {

     // --<-- plus de cas part pour l vide !!

     // on travaille avec (cur, cur->next)
      list cur = *l;
      while (cur->next != NULL) {
         cur = cur->next;
      }
      cur->next = create_cell(0);
      cur->next->val = v;
      cur->next->next = NULL;
}

/* Solution pour liste vide valant sentinelle : suppression en queue */
int remove_last_sent (list * l) {
    // précondition : *l a un élément => (*l)->next != NULL
    assert(*l != NULL);
    
    //  -- <-- plus de cas particulier
    list cur = *l;
    // cur != null et cur->next != NULL
    while (cur->next->next != NULL) {
        cur = cur->next;
    }
    // cur->next->next == NULL
    list tmp = cur->next;
    int v = tmp->val;
    cur->next = NULL;
    free(tmp);
    return v;
}

void destroy (list * l) {
    while (*l != NULL) remove_first(l);
}

int get_first (list l) {
    assert (l != NULL);
    return l-> val;
}

int get_last (list l) {
    assert (l != NULL);
    list cur = l;
    while (cur != NULL) {
        if (cur->next == NULL) break;
        cur = cur->next;
    }
    return cur->val;
}

void my_test(list * l) {
    printf("\n** tests sur liste: "); print(*l); printf("\n");

    {
        printf("inverse: "); inverse(l); print(*l); printf("\n");
        printf("back: "); inverse(l); print(*l); printf("\n");
    }
    
    {
        int v;
        if (*l != NULL) {
            v = get_first(*l);
            printf("search(%d): %d; ", v, search(v, *l));
            v = get_last(*l);
            printf("search(%d): %d; ", v, search(v, *l));
        }
        v = 0;
        printf("search(%d): %d; ", v, search(v, *l));
        v = 10;
        printf("search(%d): %d;\n", v, search(v, *l));
    }

    {
        int v = 3;
        printf("add_first(%d): ", v); add_first(v, l);
        print(*l); printf("; ");
        v = 4;
        printf("add_last_1(%d): ", v); add_last_1(v, l);
        print(*l); printf("; ");
        v = 5;
        printf("add_last_1(%d): ", v); add_last_1(v, l);
        print(*l); printf(";\n");
    }

    if (*l != NULL) {
        printf("remove_first: %d ", remove_first(l));
        print(*l); printf(";\n");
    }
    if (*l != NULL) {
        printf("remove_last_1: %d ", remove_last_1(l));
        print(*l); printf(";\n");
    }
    if (*l != NULL) {
        printf("remove_last_2: %d ", remove_last_2(l));
        print(*l); printf(";\n");
    }
    if (*l != NULL) {
        printf("remove_last_3: %d ", remove_last_3(l));
        print(*l); printf(";\n");
    }

    {
        int v;
        if (*l != NULL) {
            v = get_first(*l);
            printf("my_remove(%d): ", v); my_remove(v, l);
            print(*l); printf(";\n");
        }
        if (*l != NULL) {
            v = get_last(*l);
            printf("my_remove(%d): ", v); my_remove(v, l);
            print(*l); printf(";\n");
        }
        v = 0;
        printf("my_remove(%d): ", v); my_remove(v, l);
        print(*l); printf("; ");        
        v = 4;
        printf("my_remove(%d): ", v); my_remove(v, l);
        print(*l); printf(";\n");
        v = 100;
        printf("my_remove(%d): ", v); my_remove(v, l);
        print(*l); printf(";\n");
    }
}

int main(void) {

    {
        list l = NULL;
        my_test(&l);
        destroy(&l);
    }

    {
        list l = create_cell(1);
        my_test(&l);
        destroy(&l);
    }

    {
        list l = create_cell(1);
        l->next = create_cell(2);
        l->next->next = create_cell(3);
        l->next->next->next = create_cell(4);
        l->next->next->next->next = create_cell(5);
        l->next->next->next->next->next = create_cell(6);
        l->next->next->next->next->next->next = create_cell(7);
        my_test(&l);
        destroy(&l);
    }

    return EXIT_SUCCESS;
}
