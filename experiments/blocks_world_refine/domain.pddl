;-----------------------------------------------------------------------------
; SPDX-License-Identifier: GPL-3.0-only
; This file is part of the LogicLfD project.
; Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
; Contributor: Yan Zhang <yan.zhang@idiap.ch>
; -----------------------------------------------------------------------------

(define (domain blocks_world)

;remove requirements that are not needed
(:requirements :strips :equality :derived-predicates :disjunctive-preconditions :negative-preconditions)

(:predicates ;todo: define predicates here
    ;static predicates
    (arm ?arm)
    (block ?block)
    (table ?table)
    ; find-table-place and find-block-place
    (worldpose ?block ?X_WB)
    ; find-grasp
    (handpose ?block ?X_HB)
    ; find-table-place
    (table-support ?block ?X_WB ?table)
    ; find-block-place
    (block-support ?upperblock ?X_WU ?lowerblock ?X_WL)

    ; fluents 
    (empty ?arm)
    (inhand ?arm ?block)
    ; atpose means atworldpose, use atpose for assigning fluent states
    (atpose ?block ?X_WB) 
    (athandpose ?arm ?block ?X_HB)
    (clear ?block)
    (on-table ?block ?table)
    (on-block ?block1 ?block2)

)

(:action pick  ;off of a table
    :parameters (?arm ?block ?table ?X_WB ?X_HB)
    :precondition (and
        (arm ?arm)
        (block ?block)
        (table ?table)
        (worldpose ?block ?X_WB)
        (handpose ?block ?X_HB)

        (empty ?arm)
        (clear ?block)
        (atpose ?block ?X_WB)
        (on-table ?block ?table)
    ) 
    :effect (and
        (inhand ?arm ?block)
        (athandpose ?arm ?block ?X_HB)
        (not (atpose ?block ?X_WB))
        (not (empty ?arm))
        (not (on-table ?block ?table))
    )
    )

    (:action place ; place block on table
        :parameters (?arm ?block ?table ?X_WB ?X_HB) 
        :precondition (and
            (arm ?arm)
            (block ?block)
            (table ?table)
            (handpose ?block ?X_HB)
            (worldpose ?block ?X_WB)

            (inhand ?arm ?block)
            (not (empty ?arm))
            (athandpose ?arm ?block ?X_HB)
            (table-support ?block ?X_WB ?table)
        ) 
        :effect (and
            (not (athandpose ?arm ?block ?X_HB))
            (atpose ?block ?X_WB) 
            (empty ?arm)
            (not (inhand ?arm ?block))
            (on-table ?block ?table)
        )
    )

    (:action stack ;place block on lowerblock
        :parameters (?arm ?block ?lowerblock ?X_WB ?X_HB ?X_WL) 
        :precondition (and
            (arm ?arm)
            (block ?block)
            (block ?lowerblock)
            (not (= ?block ?lowerblock))
            (worldpose ?block ?X_WB)
            (handpose ?block ?X_HB)

            (inhand ?arm ?block)
            (not (empty ?arm))
            (athandpose ?arm ?block ?X_HB)
            (clear ?lowerblock)
            (atpose ?lowerblock ?X_WL)
            (block-support ?block ?X_WB ?lowerblock ?X_WL)
        )
        :effect (and
            (empty ?arm)
            (not (clear ?lowerblock))
            (not (athandpose ?arm ?block ?X_HB))
            (atpose ?block ?X_WB) 
            (on-block ?block ?lowerblock)
        )
    )

    (:action unstack
        :parameters (?arm ?block ?lowerblock ?X_WB ?X_HB )
        :precondition (and
            (arm ?arm)
            (block ?block)
            (block ?lowerblock)
            (not (= ?block ?lowerblock))

            (empty ?arm)
            (worldpose ?block ?X_WB)
            (handpose ?block ?X_HB)

            (clear ?block)
            (atpose ?block ?X_WB)
            (on-block ?block ?lowerblock)
        ) 
        :effect (and
            (athandpose ?arm ?block ?X_HB)
            (clear ?lowerblock)
            (not (atpose ?block ?X_WB))
            (not (empty ?arm))
            (inhand ?arm ?block)
            (not (on-block ?block ?lowerblock))
        )
    )


)