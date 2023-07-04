Toy data for testing the Monshi class.

sub: The row containing the subject ID.

scale1_X: The answer to the X-th item of "scale1."

The scoring rules for "scale1" are:
- Choices range from 1 to 5.
- Items 1 and 2 are reverse scored.
- Average score is calculated instead of a total score.
- Average is calculated by dividing the total score by the number of items minus the number of NaN answers.
- i.e., total score / (number of items - number of NaN answer)

The scoring rules for "scale2" are:
- Choices range from 1 to 6.
- Answers greater than 3 (4, 5, 6) are converted to 1; others (1, 2, 3) are converted to 0
- Items 1, 3, and 4 are reverse scored.
- Two subscales and total scores are calculated:
-- "subscale_a" including items 1, 3, 5, 7, and 9.
-- To calculate "subscale_a," the score of item 1 is not to be reverse scored.
-- "subscale_b" including items 2, 4, 6, 8, and 10.
