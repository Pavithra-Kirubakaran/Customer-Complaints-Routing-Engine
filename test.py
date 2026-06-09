from routing import route_ticket

ticket = """
My payment was deducted twice but the order was not placed.
Please refund immediately.
"""

result = route_ticket(ticket)

print(result)