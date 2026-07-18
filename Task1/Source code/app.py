import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# In-memory dictionary to track customer session states
session_state = {}

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Interactive AI Support Agent</title>
    <style>
        body { font-family: Arial; display: flex; justify-content: center; align-items: center; height: 100vh; background: #f0f2f5; margin: 0; }
        #box { width: 360px; height: 480px; background: white; border-radius: 12px; display: flex; flex-direction: column; box-shadow: 0 4px 15px rgba(0,0,0,0.1); overflow: hidden; }
        #chat { flex: 1; padding: 15px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
        .msg { padding: 10px 14px; border-radius: 15px; max-width: 75%; font-size: 14px; line-height: 1.4; }
        .user { background: #0084ff; color: white; align-self: flex-end; }
        .bot { background: #e4e6eb; color: black; align-self: flex-start; }
        #input-area { display: flex; border-top: 1px solid #ddd; }
        input { flex: 1; padding: 15px; border: none; outline: none; font-size: 14px; }
        button { padding: 15px; background: #0084ff; color: white; border: none; cursor: pointer; font-weight: bold; }
    </style>
</head>
<body>
<div id="box">
    <div id="chat"></div>
    <div id="input-area">
        <input type="text" id="inp" placeholder="Type your response here...">
        <button onclick="send()">Send</button>
    </div>
</div>
<script>
    // 🚀 STEP 1: The agent proactively speaks FIRST right as the screen opens up
    window.onload = function() {
        show("Hello! 👋 Thanks for reaching out.", 'bot');
    };

    async function send() {
        const inp = document.getElementById("inp");
        const txt = inp.value.trim();
        if(!txt) return;
        
        show(txt, 'user');
        inp.value = "";

        const res = await fetch('/webhook', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: txt })
        });
        const data = await res.json();
        data.responses.forEach(t => show(t, 'bot'));
    }

    function show(text, sender) {
        const c = document.getElementById("chat");
        const d = document.createElement("div");
        d.className = `msg ${sender}`;
        d.innerText = text;
        c.appendChild(d);
        c.scrollTop = c.scrollHeight;
    }
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/webhook", methods=["POST"])
def webhook():
    user_message = request.json.get("message", "").strip()
    user_key = "customer_1"  # Local system tracking key
    
    # Initialize state memory tracker for brand new sessions
    if user_key not in session_state:
        session_state[user_key] = "agent_greeted_first"

    current_step = session_state[user_key]

    # 🚀 STEP 3: User replied to our greeting. Agent immediately raises assistance options menu.
    if current_step == "agent_greeted_first":
        session_state[user_key] = "awaiting_menu_choice"
        return jsonify({"responses": [
            "How can I assist you with your customer account today?",
            "Please select an option by typing the number:\n1️⃣ Track your order\n2️⃣ Dissatisfied with your order"
        ]})

    # Step: Main Menu Choice Processing Logic
    elif current_step == "awaiting_menu_choice":
        if "1" in user_message or "track" in user_message.lower():
            session_state[user_key] = "awaiting_order_id"
            return jsonify({"responses": ["Please enter your 6-digit Order ID tracking number to verify your shipment status:"]})
            
        elif "2" in user_message or "dissatisfied" in user_message.lower() or "not like" in user_message.lower():
            session_state[user_key] = "awaiting_replacement_decision"
            return jsonify({"responses": ["We are very sorry to hear that you are not satisfied with your order! 😟\n\nWould you like us to issue a free replacement item? (Yes / No)"]})
            
        else:
            return jsonify({"responses": ["I didn't quite catch that. Please type '1' for Tracking or '2' if you are Dissatisfied with your item."]})

    # Step: Order Tracking Flow Endpoint
    elif current_step == "awaiting_order_id":
        session_state[user_key] = "awaiting_menu_choice"
        return jsonify({"responses": [
            "Thank you. Your order tracking status updates: Package is currently in transit with FedEx and will arrive within 2-3 business days.",
            "Is there anything else I can assist you with today?\n1️⃣ Track another order\n2️⃣ Dissatisfied with an order"
        ]})

    # Step: Product Dissatisfaction Flow Endpoint
    elif current_step == "awaiting_replacement_decision":
        if any(word in user_message.lower() for word in ["yes", "yeah", "yep", "replace"]):
            session_state[user_key] = "awaiting_menu_choice"
            return jsonify({"responses": [
                "Perfect! We have successfully submitted a zero-cost replacement request. A new package tracking link will hit your inbox shortly! 📦",
                "Need anything else?\n1️⃣ Track your order\n2️⃣ Dissatisfied with an order"
            ]})
            
        elif any(word in user_message.lower() for word in ["no", "nope", "refund"]):
            session_state[user_key] = "awaiting_refund_confirmation"
            return jsonify({"responses": ["Understood. Since a replacement won't work, would you prefer a complete refund sent back to your original payment method? (Yes / No)"]})
            
        else:
            return jsonify({"responses": ["Please type 'Yes' to request a replacement item or 'No' to look at other choices."]})

    # Step: Final Refund Choice Processing Logic
    elif current_step == "awaiting_refund_confirmation":
        session_state[user_key] = "awaiting_menu_choice"
        if any(word in user_message.lower() for word in ["yes", "yeah", "yep", "refund"]):
            return jsonify({"responses": [
                "Success! 💸 Your full refund request has been processed. The funds will show up on your banking statement within 3 to 5 business days.",
                "How can I help you next?\n1️⃣ Track your order\n2️⃣ Dissatisfied with an order"
            ]})
        else:
            return jsonify({"responses": [
                "No problem. We have canceled the return request. Your support session has been reset.",
                "How can I help you next?\n1️⃣ Track your order\n2️⃣ Dissatisfied with an order"
            ]})

    return jsonify({"responses": ["Session error. Let's start over."]})

if __name__ == "__main__":
    app.run(port=5000, debug=True)