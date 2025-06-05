from SAEVs_calculation import *

data = pd.read_excel('inputs.xlsx')
model = solve_opt(data,pass_min=2000, mult_2=0,tee=True)

# Plot the results
D_passenger = (data['Total demand'][:96]).tolist()
O_energy = (data['Outage'][:96]).tolist()
#plot_variables(model, D_passenger, O_energy)

# Save results to Excel
filename = 'test.xlsx'
#save_to_excel(model, filename=filename)