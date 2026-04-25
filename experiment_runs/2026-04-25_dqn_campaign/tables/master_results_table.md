# Master Results Table

## DQN Long-Run Training Summary

| Mode   | Episodes | Reward Early | Reward Late | Improving | Satisfaction (last) | Complaints (last) | HP Satisfaction (last) | Final Loss | Final Epsilon | Buffer | Steps | Status |
|--------|----------|--------------|-------------|-----------|----------------------|-------------------|------------------------|------------|---------------|--------|-------|--------|
| Easy   | 1000     | 27.332       | 42.526      | True      | 8.89                 | 0.00              | 1.94                   | 0.1643     | 0.0496        | 15000  | 49001 | Complete |
| Medium | 1000     | 43.584       | 55.117      | True      | 8.88                 | 8.98              | 2.93                   | 0.2978     | 0.0200        | 20000  | 48501 | Complete |
| Hard   | 1000     | 14.636       | 28.571      | True      | 7.38                 | 40.67             | 3.24                   | 0.3227     | 0.0200        | 25000  | 48001 | Complete |

## Notes
- Medium is the strongest result overall.
- Easy is highly stable with zero complaints.
- Hard also shows clear long-run learning, though it remains the most difficult benchmark.
