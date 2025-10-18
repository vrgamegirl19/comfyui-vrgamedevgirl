import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Create popup notification system
class VRGDGNotification {
    constructor() {
        this.container = null;
        this.activePopups = [];
        this.init();
    }

    init() {
        // Create container for popups - TOP CENTER
        this.container = document.createElement('div');
        this.container.id = 'vrgdg-popup-container';
        this.container.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-width: 500px;
        `;
        document.body.appendChild(this.container);

        // Listen for popup messages
        api.addEventListener("vrgdg_instructions_popup", (event) => {
            const data = event.detail;
            this.show(data.message, data.type || 'info', data.title || 'Notification');
        });
    }

    show(message, type = 'info', title = 'Notification') {
        const popup = document.createElement('div');
        popup.className = `vrgdg-popup vrgdg-popup-${type}`;
        
        // Enhanced color scheme
        const colors = {
            info: { bg: '#2196F3', border: '#1976D2' },           // Blue
            warning: { bg: '#FF9800', border: '#F57C00' },        // Orange
            error: { bg: '#F44336', border: '#D32F2F' },          // Dark Red
            success: { bg: '#4CAF50', border: '#388E3C' },        // Dark Green
            red: { bg: '#DC3545', border: '#C82333' },            // Bright Red
            yellow: { bg: '#FFC107', border: '#FFA000' },         // Yellow
            green: { bg: '#28A745', border: '#218838' }           // Bright Green
        };
        
        const color = colors[type] || colors.info;
        
        popup.style.cssText = `
            background: ${color.bg};
            color: white;
            padding: 20px 24px;
            border-radius: 8px;
            border-left: 5px solid ${color.border};
            box-shadow: 0 6px 16px rgba(0,0,0,0.4);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            line-height: 1.6;
            max-width: 500px;
            min-width: 400px;
            word-wrap: break-word;
            animation: slideDown 0.4s ease-out;
            position: relative;
            cursor: default;
        `;

        // Title
        const titleEl = document.createElement('div');
        titleEl.style.cssText = `
            font-weight: bold;
            font-size: 17px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        `;
        
        const titleText = document.createElement('span');
        titleText.textContent = title;
        
        // Close button
        const closeBtn = document.createElement('button');
        closeBtn.innerHTML = '✕';
        closeBtn.style.cssText = `
            background: none;
            border: none;
            color: white;
            font-size: 22px;
            cursor: pointer;
            padding: 0;
            margin-left: 12px;
            opacity: 0.8;
            transition: opacity 0.2s;
            line-height: 1;
        `;
        closeBtn.onmouseover = () => closeBtn.style.opacity = '1';
        closeBtn.onmouseout = () => closeBtn.style.opacity = '0.8';
        closeBtn.onclick = () => this.close(popup);
        
        titleEl.appendChild(titleText);
        titleEl.appendChild(closeBtn);

        // Message (preserve formatting)
        const messageEl = document.createElement('div');
        messageEl.style.cssText = `
            white-space: pre-wrap;
            font-size: 14px;
        `;
        messageEl.textContent = message;

        popup.appendChild(titleEl);
        popup.appendChild(messageEl);
        
        this.container.appendChild(popup);
        this.activePopups.push(popup);

        // Auto-close after 20 seconds
        setTimeout(() => {
            if (popup.parentNode) {
                this.close(popup);
            }
        }, 20000);
    }

    close(popup) {
        popup.style.animation = 'slideUp 0.3s ease-in';
        setTimeout(() => {
            if (popup.parentNode) {
                this.container.removeChild(popup);
                this.activePopups = this.activePopups.filter(p => p !== popup);
            }
        }, 300);
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideDown {
        from {
            transform: translateY(-100px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes slideUp {
        from {
            transform: translateY(0);
            opacity: 1;
        }
        to {
            transform: translateY(-100px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize on app setup
app.registerExtension({
    name: "VRGDG.PopupNotifications",
    async setup() {
        window.vrgdgNotification = new VRGDGNotification();
        console.log("[VRGDG] Popup notification system loaded");
    }
});