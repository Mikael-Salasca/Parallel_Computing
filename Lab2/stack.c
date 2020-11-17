/*
 * stack.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 *
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif

struct cell_t * create_cell (int v) {
    cell_t *  new;
    new = malloc(sizeof(struct cell));
    new->val = v;
    new->next = NULL;
    return new;
}


int
stack_check(stack_t *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
	// Use assert() to check if your stack is in a state that makes sens
	// This test should always pass
	assert(1 == 1);
	// This test fails if the task is not allocated or if the allocation failed
	assert(stack != NULL);
#endif
	// The stack is always fine
	return 1;
}

int /* Return the type you prefer */
stack_push(stack_t * s, int val)
{
	assert(s != NULL);
	cell_t* old;
	cell_t* new_cell = create_cell(val);


#if NON_BLOCKING == 0
  // Implement a lock_based stack
	pthread_mutex_lock(&mtx);
	new_cell->next = s->head;
  s->head = new_cell;
  pthread_mutex_unlock(&mtx);


#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
	do {
	    old = s->head;
	    new_cell->next = old;
	  } while(cas((size_t*)&s->head, (size_t)old, (size_t)new_cell) != (size_t)old);



#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check((stack_t*)1);

  return 0;
}

int /* Return the type you prefer */
stack_pop(stack_t* s) {
#if NON_BLOCKING == 0
  // Implement a lock_based stack
  cell_t* to_remove;

  pthread_mutex_lock(&mtx);
    to_remove = s->head;
    s->head = s->head->next;
  pthread_mutex_unlock(&mtx);

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  struct node* old;

  // removes the element from the stack
  do {
      old = s->head;
  } while(cas((size_t*)&s->head, (size_t)old, (size_t)old->next) != (size_t)old);

#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  return 0;
}
