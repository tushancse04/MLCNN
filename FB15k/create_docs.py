import xlrd 

loc = ("Report.xlsx") 
  
# To open Workbook 
wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0) 

for i in range(36):
	name = sheet.cell_value(i+1, 0) +  sheet.cell_value(i+1, 1)
	print(name)