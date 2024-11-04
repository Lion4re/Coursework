/*
 * ============================================
 * file: streaming_service.c
 * @Author Vasileios Papageridis (csd4710@csd.uoc.gr)
 * @Version 23/10/2023
 *
 * @e-mail hy240@csd.uoc.gr
 *
 * @brief Streaming_service function
 *        for CS240 Project Phase 1,
 *        Winter Semester 2023-2024
 * @see   Compile using supplied Makefile by running: make
 * ============================================
 */
#include <stdio.h>
#include <stdlib.h>

#include "streaming_service.h"

/* Category table */
struct movie *category_talbe[6];

/* Head of users-list */
struct user *usersHead;

/* Sentinel node pointer */
struct user *sentinel;

/* Head of New Releases List */
struct new_movie *newMoviesHead;

/* Function to delete New Releases list linked list */
void deleteNewRealeasesList(struct new_movie **head){
	struct new_movie *current = *head;
	struct new_movie *next = NULL;
	
	while(current != NULL){
		next = current->next;
		free(current);
		current = next;
	}
	*head = NULL;
}

/* Function to delete Movie list linked list */
void deleteMovieList(struct movie **head){
	struct movie *current = *head;
	struct movie *next = NULL;
	
	while(current != NULL){
		next = current->next;
		free(current);
		current = next;
	}
	*head = NULL;
}

/* Function to delete a Suggested Movie List */
void deleteSuggestedMoviesList(struct suggested_movie **head){
    struct suggested_movie *current = *head;

    /* Empty the doubly linked list of recommended movies */
    struct suggested_movie *temp = NULL;
    while(current != NULL){
        temp = current;
        current = current->next;

        free(temp);
        temp = NULL;
    }
    *head = NULL;
    
}

/* Function to delete a Watch History Stack */
void deleteWatchHistoryStack(struct movie **head){
    struct movie *temp = NULL;

    while(*head != NULL){
        temp = *head;
        *head = (*head)->next;

        free(temp);
        temp = NULL;
    }
    *head = NULL;
}

/* Function to delete users list linked list */
void deleteUsersList(struct user **head){
	struct user *current = *head;
	struct user *next = NULL;
	
	while(current != NULL){
		next = current->next;
        
        deleteSuggestedMoviesList(&(current->suggestedHead));
        deleteWatchHistoryStack(&(current->watchHistory));

		free(current);
		current = next;
	}
	*head = NULL;
}

/* Function to push a movie to the watch history stack */
void push(struct movie **watchHistoryTop ,unsigned mid, unsigned year){
    struct movie *new_movie = (struct movie *) malloc(sizeof(struct movie));
    if(new_movie == NULL){
        printf("Memory not allocated. malloc() fail...\n");
        return;
    }

    new_movie->info.mid = mid;
    new_movie->info.year = year;

    new_movie->next = *watchHistoryTop;
    *watchHistoryTop = new_movie;
    
}

/* Function to pop a movie from the watch history stack */
struct movie* pop(struct user *user){
    struct movie *temp = NULL;

    temp = user->watchHistory;
    user->watchHistory = user->watchHistory->next;
    return temp;
}

/* Function that returns if the watch history stack is empty */
int isEmpty(struct movie *watchHistoryTop){
    if(watchHistoryTop == NULL){
        return 1;
    }
    else{
        return 0;
    }
}

/* Function to print the watch history stack */
void printWatchHistory(struct movie *watchHistoryTop ,int uid, unsigned mid){
    struct movie *temp = NULL;
    
    if(isEmpty(watchHistoryTop)){
        printf("Watch History stack is empty.\n");
        return;
    }

    printf("\nW uid:<%d>, mid:<%d>\n", uid, mid);
    printf("\tUser <%d> : Watch History = ", uid);
    temp = watchHistoryTop;
    while(temp != NULL){
        printf(" %d |", temp->info.mid);
        temp = temp->next;
    }
    printf("\nDONE\n");
}

int is_movie_watched(struct movie *watchHistoryHead, int mid) {
    struct movie *current = watchHistoryHead;
    while(current != NULL) {
        if(current->info.mid == mid) {
            return 1;
        }
        current = current->next;
    }
    return 0;
}

/*
 * Sorted Movie Inster
 *
 * Helpful function to instert a movie
 * in a sorted list (ascending order)
 * 
 * 
 * 
 */
void sortedMovieInstert(struct movie ** headref, struct movie *movieNode){
    struct movie *current = NULL;

    if((*headref) == NULL || (*headref)->info.mid >= movieNode->info.mid){
        movieNode->next = (*headref);
        (*headref) = movieNode;
    }
    else{
        current = (*headref);
        while(current->next != NULL && current->next->info.mid < movieNode->info.mid){
            current = current->next;
        }

        movieNode->next = current->next;
        current->next = movieNode;
    }
}

void sortedNewMovieInstert(struct new_movie ** headref, struct new_movie *movieNode){
    struct new_movie *current = NULL;

    if((*headref) == NULL || (*headref)->info.mid >= movieNode->info.mid){
        movieNode->next = (*headref);
        (*headref) = movieNode;
    }
    else{
        current = (*headref);
        while(current->next != NULL && current->next->info.mid < movieNode->info.mid){
            current = current->next;
        }

        movieNode->next = current->next;
        current->next = movieNode;
    }
}

/* Function to insert as next in doubly linked list */
void insertAsNext(struct suggested_movie **front, struct movie *movie, struct suggested_movie **head){
    struct suggested_movie * movieToSuggest = NULL;

    movieToSuggest = (struct suggested_movie *) malloc(sizeof(struct suggested_movie));
    if(movieToSuggest == NULL){
        printf("Memory not allocated. malloc() fail...\n");
        return;
    }

    movieToSuggest->info.mid = movie->info.mid;
    movieToSuggest->info.year = movie->info.year;

    if((*head) == NULL){
        movieToSuggest->prev = NULL;
        movieToSuggest->next = NULL;
        (*head) = movieToSuggest;
        (*front) = movieToSuggest;
    }
    else{
        movieToSuggest->prev = (*front);
        movieToSuggest->next = NULL;
        (*front)->next = movieToSuggest;
        (*front) = movieToSuggest;
    }

}

/* Function to insert as previous in doubly linked list */
void insertAsPrev(struct suggested_movie **back, struct movie *movie, struct suggested_movie **tail){
    struct suggested_movie *movieToSuggest = NULL;

    movieToSuggest = (struct suggested_movie *) malloc(sizeof(struct suggested_movie));
    if(movieToSuggest == NULL){
        printf("Memory not allocated. malloc() fail...\n");
        return;
    }

    movieToSuggest->info.mid = movie->info.mid;
    movieToSuggest->info.year = movie->info.year;

    if((*tail) == NULL){
        movieToSuggest->next = NULL;
        movieToSuggest->prev = NULL;
        (*back) = movieToSuggest;
        (*tail) = movieToSuggest;
    }
    else{
        movieToSuggest->next = (*back);
        movieToSuggest->prev = NULL;
        (*back)->prev = movieToSuggest;
        (*back) = movieToSuggest;
    }
}

/* Function to insert a new movie with sorted way in doubly linked list of suggested movies */
void sortedSuggestMovieInsert(struct suggested_movie **head, struct suggested_movie **tail, struct suggested_movie *new_movie){
    struct suggested_movie *current = NULL;
    
    if(*head == NULL){
        *head = new_movie;
        *tail = new_movie;
    }
    else if((*head)->info.mid >= new_movie->info.mid){
        new_movie->next = *head;
        new_movie->next->prev = new_movie;
        *head = new_movie;
    }
    else{
        current = *head;

        while(current->next != NULL && current->next->info.mid < new_movie->info.mid){
            current = current->next;
        }

        new_movie->next = current->next;

        if(current->next != NULL){
            new_movie->next->prev = new_movie;
        }
        else{
            *tail = new_movie;  /* Update tail if new_movie is at the end */
        }

        current->next = new_movie;
        new_movie->prev = current;
    }
}


/* Function that links 2 doubly linked lists */
void linkLists(struct suggested_movie **front, struct suggested_movie **back){
    if(*front != NULL && *back != NULL){
        (*front)->next = (*back);
        (*back)->prev = (*front);
    }
}

/* Function that prints all the lists of the Category Table */
void printCategoryTable(){
    struct movie *current = NULL;
    int i;

    printf("Categorized Movies:");
    for(i = 0; i < 6; i++){
    current = category_talbe[i];
    
    switch(i){
        case 0:
            printf("\n0.\tHorror:");
            while(current != NULL){
                printf(" %d |", current->info.mid);
                current = current->next;
            }
            break;
        case 1:
            printf("\n1.\tSci-Fi:");
            while(current != NULL){
                printf(" %d |", current->info.mid);
                current = current->next;
            }
            break;
        case 2:
            printf("\n2.\tDrama:");
            while(current != NULL){
                printf(" %d |", current->info.mid);
                current = current->next;
            }
            break;
        case 3:
            printf("\n3.\tRomance:");
            while(current != NULL){
                printf(" %d |", current->info.mid);
                current = current->next;
            }
            break;
        case 4:
            printf("\n4.\tDocumentary:");
            while(current != NULL){
                printf(" %d |", current->info.mid);
                current = current->next;
            }
            break;
        case 5:
            printf("\n5.\tComedy:");
            while(current != NULL){
                printf(" %d |", current->info.mid);
                current = current->next;
            }
            break;
    }
}
}

/*  Print users list
    Prints the users list, after the registration (R) or unregistration (U) of a user
*/
void print_users_list(struct user *head, char command, int uid){
    struct user *current = head;

    printf("\n%c uid:<%d>\n", command, uid);
    printf("\tUsers =");
    while(current != sentinel){
        printf(" %d |", current->uid);
        current = current->next;
    }
    printf("\nDONE\n");
}

/*
 * Register User - Event R
 * 
 * Adds user with ID uid to
 * users list, as long as no
 * other user with the same uid
 * already exists.
 *
 * Returns 0 on success, -1 on
 * failure (user ID already exists,
 * malloc or other error)
 */
int register_user(int uid){
    struct user *current = usersHead;

    /* Checking that user is not already in the list */
    while(current != sentinel){
        if(current->uid == uid){
            printf("\nUser %d already exists\n", uid);
            return -1;
        }
        current = current->next;
    }

    /* Allocating memory for the new user node */
    struct user *new_user = (struct user *) malloc(sizeof(struct user));
    if(new_user == NULL){
        printf("Memory not allocated. malloc() fail...\n");
        return -1;
    }

    /* Initializing the new user node */
    new_user->uid = uid;
    new_user->suggestedHead = NULL;
    new_user->suggestedTail = NULL;
    new_user->watchHistory = NULL;
    new_user->next = usersHead;

    /* The head now points to the new user */
    usersHead = new_user;

    /* Print the users list after the registration */
    print_users_list(usersHead, 'R', uid);

    return 0;
}

/*
 * Unregister User - Event U
 *
 * Removes user with ID uid from
 * users list, as long as such a
 * user exists, after clearing the
 * user's suggested movie list and
 * watch history stack
 */
void unregister_user(int uid){
    struct user *current = usersHead;
    struct user *previous = NULL;
    int flag = 0;

    /* Traverse the users list to find the user */
    while(current != sentinel){
        if(current->uid == uid){
            flag = 1;   /* indicates that user found */
            break;
        }
        previous = current;
        current = current->next;
    }

    /* Checks if user not found in the list */
    if(flag == 0){
        printf("User not found.\n");
        return;
    }

    /* Empty the doubly linked list of recommended movies */
    deleteSuggestedMoviesList(&(current->suggestedHead));

    /* Empty the watch history stack */
    deleteWatchHistoryStack(&(current->watchHistory));
    

    if(previous == NULL){
        printf("List is empty\n");
    }
    else{
        previous->next = current->next;
        
        free(current);
        current = NULL;
    }

    print_users_list(usersHead, 'U', uid);
}

/*
 * Add new movie - Event A
 *
 * Adds movie with ID mid, category
 * category and release year year
 * to new movies list. The new movies
 * list must remain sorted (increasing
 * order based on movie ID) after every
 * insertion.
 *
 * Returns 0 on success, -1 on failure
 */
int add_new_movie(unsigned mid, movieCategory_t category, unsigned year){
    struct new_movie *current = newMoviesHead;

    /* Traverese the list to check if the movie is already in the list */
    while(current != NULL){
        if(current->info.mid == mid){
            printf("Movie already exists in New Releases List\n");
            return -1;
        }
        current = current->next;
    }

    /* Allocate memory for the new movie and assign it's fields */
    struct new_movie *new_release = (struct new_movie *) malloc(sizeof(struct new_movie));
    if(new_release == NULL){
        printf("Memory not allocated. malloc() fail...\n");
        return -1;
    }
    new_release->info.mid = mid;
    new_release->info.year = year;
    new_release->category = category;

    /* Insert the new release in the new releases list */
    sortedNewMovieInstert(&newMoviesHead, new_release);

    /* Print of the event A */
    printf("\nA: Mid:<%d> \t Category:<%d> \t Year:<%d>\n", mid, category, year);
    printf("New movies =");
    current = newMoviesHead;
    while(current != NULL){
        printf(" %d, %d, %d |", current->info.mid, current->category, current->info.year);
        current = current->next;
    }
    printf("\nDONE\n");

    return 0;
}


/*
 * Distribute new movies - Event D
 *
 * Distributes movies from the new movies
 * list to the per-category sorted movie
 * lists of the category list array. The new
 * movies list should be empty after this
 * event. This event must be implemented in
 * O(n) time complexity, where n is the size
 * of the new movies list
 */
void distribute_new_movies(void){
    struct new_movie *current = newMoviesHead;
    struct movie *current2 = NULL;
    struct new_movie *toDelete = NULL;
    struct movie *distr_movie = NULL;
    int i;

    /* If list is empty, then return */
    if(newMoviesHead == NULL){
        printf("\nNo movies to distribute, New Releases List is empty.\n");
        return;
    }

    while(newMoviesHead != NULL){
        distr_movie = (struct movie *) malloc(sizeof(struct movie));
        if(distr_movie == NULL){
            printf("Memory not allocated. malloc() fail...\n");
            return;
        }
        distr_movie->info.mid = newMoviesHead->info.mid;
        distr_movie->info.year = newMoviesHead->info.year;

        /* Insert the movie based on the category */
        switch(newMoviesHead->category){
            case HORROR:
                sortedMovieInstert(&category_talbe[HORROR], distr_movie);
                break;

            case SCIFI:
                sortedMovieInstert(&category_talbe[SCIFI], distr_movie);
                break;

            case DRAMA:
                sortedMovieInstert(&category_talbe[DRAMA], distr_movie);
                break;

            case ROMANCE:
                sortedMovieInstert(&category_talbe[ROMANCE], distr_movie);
                break;

            case DOCUMENTARY:
                sortedMovieInstert(&category_talbe[DOCUMENTARY], distr_movie);
                break;

            case COMEDY:
                sortedMovieInstert(&category_talbe[COMEDY], distr_movie);
                break;
        }

        /* Delete the movie from the New Releases after the insertion in the Category table */
        toDelete = newMoviesHead;
        newMoviesHead = newMoviesHead->next;
        free(toDelete);
    }

    printf("\nD\n");
    printCategoryTable();
    printf("\nDONE\n");
}

/*
 * User watches movie - Event W
 *
 * Adds a new struct movie with information
 * corresponding to those of the movie with ID
 * mid to the top of the watch history stack
 * of user uid.
 *
 * Returns 0 on success, -1 on failure
 * (user/movie does not exist, malloc error)
 */
int watch_movie(int uid, unsigned mid){
    int i;
    int movieFlag = 0;
    struct movie *currentMovie = NULL;
    struct user *currentUser = NULL;

    /* Searching for the movie by traversing each list in the Category table */
    for(i = 0; i < 6; i++){
        currentMovie = category_talbe[i];
        while(currentMovie != NULL){
            if(currentMovie->info.mid == mid){
                movieFlag = 1;  /* Indicates that the movie was found */
                break;
            }
            currentMovie = currentMovie->next;
        }
        /* If movie found then exit the loop */
        if(movieFlag == 1){
            break;
        }
    }

    /* Searching for the user by traversing the users list */
    currentUser = usersHead;
    while(currentUser != sentinel){
        if(currentUser->uid == uid){
            break;
        }
        currentUser = currentUser->next;
    }

    if(movieFlag = 0){
        printf("Movie not found\n");
        return -1;
    }
    else if(currentUser->uid == -1){
        printf("User not found\n");
        return -1;
    }
    else{
        push(&(currentUser->watchHistory), currentMovie->info.mid, currentMovie->info.year);
        printWatchHistory(currentUser->watchHistory, uid, mid);
        return 0;
    }
}

/*
 * Suggest movies to user - Event S
 *
 * For each user in the users list with
 * id != uid, pops a struct movie from the
 * user's watch history stack, and adds a
 * struct suggested_movie to user uid's
 * suggested movies list in alternating
 * fashion, once from user uid's suggestedHead
 * pointer and following next pointers, and
 * once from user uid's suggestedTail pointer
 * and following prev pointers. This event
 * should be implemented with time complexity
 * O(n), where n is the size of the users list
 *
 * Returns 0 on success, -1 on failure
 */
int suggest_movies(int uid){
    struct user *userToSuggest = NULL;
    struct suggested_movie *currentFront = NULL;
    struct suggested_movie *currentBack = NULL;
    struct movie *movieToSuggest_tmp = NULL;
    struct user *current = usersHead;
    int counter;

    /* Traverse the list to find the user with id uid */
    while(current != sentinel){
        if(current->uid == uid){
            userToSuggest = current;
            break;
        }
        current = current->next;
    }

    /* If we hit the sentinel node user not found */
    if(current->uid == -1){
        printf("User %d not found\n", uid);
        return -1;
    }

    currentFront = userToSuggest->suggestedHead;
    currentBack = userToSuggest->suggestedTail;
    current = usersHead;
    counter = 1;
    while(current != sentinel){
        if(current->uid != uid && current->watchHistory != NULL){
            movieToSuggest_tmp = pop(current);
            if(movieToSuggest_tmp == NULL){
                return -1;
            }
            if(is_movie_watched(userToSuggest->watchHistory, movieToSuggest_tmp->info.mid)) {
                free(movieToSuggest_tmp);
                continue;
            }
            struct suggested_movie *newSuggestion = malloc(sizeof(struct suggested_movie));
            newSuggestion->info.mid = movieToSuggest_tmp->info.mid;
            newSuggestion->info.year = movieToSuggest_tmp->info.year;
            if((counter % 2) != 0){
                if(currentFront != NULL){
                    newSuggestion->next = currentFront->next;
                    if(currentFront->next != NULL){
                        currentFront->next->prev = newSuggestion;
                    }
                    currentFront->next = newSuggestion;
                    newSuggestion->prev = currentFront;
                    if(newSuggestion->next == NULL){
                        userToSuggest->suggestedTail = newSuggestion;
                    }
                    currentFront = newSuggestion;
                } else {
                    userToSuggest->suggestedHead = newSuggestion;
                    userToSuggest->suggestedTail = newSuggestion;
                    currentFront = newSuggestion;
                }
            }
            else{
                if(currentBack != NULL){
                    newSuggestion->next = currentBack;
                    newSuggestion->prev = currentBack->prev;
                    if(currentBack->prev != NULL){
                        currentBack->prev->next = newSuggestion;
                    }
                    currentBack->prev = newSuggestion;
                    if(newSuggestion->prev == NULL){
                        userToSuggest->suggestedHead = newSuggestion;
                    }
                    currentBack = newSuggestion;
                } else {
                    userToSuggest->suggestedHead = newSuggestion;
                    userToSuggest->suggestedTail = newSuggestion;
                    currentBack = newSuggestion;
                }
            }
            free(movieToSuggest_tmp);
            counter++;
        }
        current = current->next;
    }
    
    /* Print of the event S */
    printf("\nS uid:<%d>\n", userToSuggest->uid);
    printf("\tUser <%d> Suggested Movies = ", userToSuggest->uid);
    currentFront = userToSuggest->suggestedHead;
    while(currentFront != NULL){
        printf(" %d |", currentFront->info.mid);
        currentFront = currentFront->next;
    }
    printf("\nDONE\n");

    return 0;
}

/*
 * Filtered movie search - Event F
 *
 * User uid asks to be suggested movies
 * belonging to either category1 or category2
 * and with release year >= year. The resulting
 * suggested movies list must be sorted with
 * increasing order based on movie ID (as the
 * two category lists). This event should be
 * implemented with time complexity O(n + m),
 * where n, m are the sizes of the two category lists
 *
 * Returns 0 on success, -1 on failure
 */
int filtered_movie_search(int uid, movieCategory_t category1, movieCategory_t category2, unsigned year){
    int i;
    struct user *currentUser = NULL;
    struct movie *current_movie = NULL;
    struct suggested_movie *found_movie = NULL;
    struct suggested_movie *new_list = NULL;
    struct suggested_movie *new_list_tail = NULL;
    struct suggested_movie *current_sugg = NULL;

    /* In the category table searching for the 2 categories movie lists and then traversing ONLY those 2 */
    for(i = 0; i < 6; i++){
        if(i == category1 || i == category2){
            current_movie = category_talbe[i];

            while(current_movie != NULL){
                if(current_movie->info.year >= year){
                    found_movie = (struct suggested_movie *) malloc(sizeof(struct suggested_movie));
                    if(found_movie == NULL){
                        printf("malloc failed...\n");
                        return -1;
                    }

                    found_movie->info.mid = current_movie->info.mid;
                    found_movie->info.year = current_movie->info.year;
                    found_movie->next = NULL;
                    found_movie->prev = NULL;
                    
                    sortedSuggestMovieInsert(&new_list, &new_list_tail, found_movie);
                }
                current_movie = current_movie->next;
            }
        }
    }
    
    /* Searching for the user to suggest the list above */
    currentUser = usersHead;
    while(currentUser != sentinel){
        if(currentUser->uid == uid){
            break;
        }
        currentUser = currentUser->next;
    }

    if(currentUser->uid == -1){
        printf("User %d not found..\n", uid);
        return -1;
    }

    if(currentUser->suggestedHead == NULL){
        currentUser->suggestedHead = new_list;
        currentUser->suggestedTail = new_list_tail;
    }
    else{
        currentUser->suggestedTail->next = new_list;
        new_list->prev = currentUser->suggestedTail;
    }

    printf("\nF uid:<%d>,\t 1st Category:<%d>, 2nd Category:<%d> \t Year:<%d>\n", currentUser->uid, category1, category2, year);
    current_sugg = currentUser->suggestedHead;

    printf("\tUser <%d> Suggested Movies = ", currentUser->uid);
    while(current_sugg != NULL){
        printf(" %d |", current_sugg->info.mid);
        current_sugg = current_sugg->next;
    }
    printf("\nDONE\n");

    return 0;
}

/*
 * Take off movie - Event T
 *
 * Movie mid is taken off the service. It is removed
 * from every user's suggested list -if present- and
 * from the corresponding category list.
 */
void take_off_movie(unsigned mid){
    struct user *currentUser = NULL;
    struct suggested_movie *currentSuggested = NULL;
    struct suggested_movie *prevSuggested = NULL;
    struct suggested_movie *toDelete = NULL;
    struct movie *currentMovie = NULL;
    struct movie *previousMovie = NULL;
    
    int i;
    int flag;

    printf("\nT mid<%d>\n", mid);
    currentUser = usersHead;
    while(currentUser != sentinel){
        currentSuggested = currentUser->suggestedHead;
        while(currentSuggested != NULL){
            if(currentSuggested->info.mid == mid){
                if(currentUser->suggestedHead == NULL || currentSuggested == NULL){
                    return;
                }
                
                if(currentUser->suggestedHead == currentSuggested){
                    currentUser->suggestedHead = currentSuggested->next;
                }

                if(currentSuggested->next != NULL){
                    currentSuggested->next->prev = currentSuggested->prev;
                }

                if(currentSuggested->prev != NULL){
                    currentSuggested->prev->next = currentSuggested->next;
                }

                printf("\tMovie:<%d> removed from uid:<%d> Suggested list\n", mid, currentUser->uid);
                toDelete = currentSuggested;
                currentSuggested = currentSuggested->next;
                if(prevSuggested != NULL){
                    prevSuggested->next = currentSuggested;
                }
                free(toDelete);
            } else {
                prevSuggested = currentSuggested;
                currentSuggested = currentSuggested->next;
            }
        }
        currentUser = currentUser->next;
    }

    for(i = 0; i < 6; i++){
        currentMovie = category_talbe[i];
        flag = 0;
        while(currentMovie != NULL){
            if(currentMovie->info.mid == mid){
                flag = 1;
                if(currentMovie == category_talbe[i]){
                    category_talbe[i] = currentMovie->next; /* Make the next node the first node */
                }
                else{
                    if(previousMovie != NULL){
                        previousMovie->next = currentMovie->next;
                    }
                }
                printf("\tMovie:<%d> removed from Category:<%d> list\n", mid, i);
                free(currentMovie);
                break;
            }
            previousMovie = currentMovie;
            currentMovie = currentMovie->next;
        }

        if(flag == 1){
            currentMovie = category_talbe[i];
            printf("\tCategory list %d = ", i);
            while(currentMovie != NULL){
                printf(" %d |", currentMovie->info.mid);
                currentMovie = currentMovie->next;
            }
        }
    }
    printf("\nDONE\n");
}

/*
 * Print movies - Event M
 *
 * Prints information on movies in
 * per-category lists
 */
void print_movies(void){
    printf("\nM\n");
    printCategoryTable();
    printf("\nDONE\n");
}

/*
 * Print users - Event P
 *
 * Prints information on users in
 * users list
 */
void print_users(void){
    struct user *currnetUser = NULL;
    struct suggested_movie *currentSuggest = NULL;
    struct movie *currentMovie = NULL;

    printf("\nP\nUsers:");

    currnetUser = usersHead;
    while(currnetUser != sentinel){
        printf("\nuid:<%d>\n", currnetUser->uid);

        currentSuggest = currnetUser->suggestedHead;
        printf("\tSuggested: ");
        while(currentSuggest != NULL){
            printf(" %d |", currentSuggest->info.mid);
            currentSuggest = currentSuggest->next;
        }

        currentMovie = currnetUser->watchHistory;
        printf("\n\tWatch History:");
        while(currentMovie != NULL){
            printf(" %d |", currentMovie->info.mid);
            currentMovie = currentMovie->next;
        }
        printf("\n");
        currnetUser = currnetUser->next;
    }

    printf("\nDONE\n");
}