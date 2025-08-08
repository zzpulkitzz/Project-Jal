import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  Droplets, 
  Waves, 
  MessageCircle, 
  Sparkles,
  FileText,
  Upload,
  AlertCircle,
  CheckCircle,
  Loader2
} from 'lucide-react';
import axios from 'axios';

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm AquaBot, your water conservation companion. I'm here to help you learn about water-saving techniques, best practices for clean water, and the importance of sanitation. How can I help you today?",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState('disconnected');
  const [isInitialized, setIsInitialized] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    checkSystemStatus();
  }, []);

  const checkSystemStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      setSystemStatus(response.data.ready ? 'ready' : 'initialized');
      setIsInitialized(response.data.ready);
    } catch (error) {
      setSystemStatus('disconnected');
    }
  };

  const initializeWithSampleData = async () => {
    try {
      setIsLoading(true);
      // This would typically point to your water conservation documents
      const response = await axios.post(`${API_BASE_URL}/initialize`, {
        file_paths: ["/path/to/water-conservation-guide.txt"] // Update this path
      });
      
      if (response.data.ready) {
        setIsInitialized(true);
        setSystemStatus('ready');
        addMessage("Great! I'm now loaded with water conservation knowledge and ready to help you!", 'bot');
      }
    } catch (error) {
      addMessage("Sorry, I couldn't load my knowledge base. Please check if the backend is running and documents are available.", 'bot');
    } finally {
      setIsLoading(false);
    }
  };

  const addMessage = (text, sender, sources = []) => {
    const newMessage = {
      id: Date.now(),
      text,
      sender,
      timestamp: new Date(),
      sources
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage = inputText.trim();
    setInputText('');
    addMessage(userMessage, 'user');

    if (!isInitialized) {
      addMessage("I need to initialize my knowledge base first. Let me do that for you!", 'bot');
      await initializeWithSampleData();
      return;
    }

    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/query`, {
        question: userMessage
      });

      const { answer, sources } = response.data;
      addMessage(answer, 'bot', sources);
    } catch (error) {
      let errorMessage = "I'm sorry, I encountered an error while processing your question. ";
      
      if (error.response?.status === 503) {
        errorMessage += "My knowledge base isn't initialized yet. Let me set that up!";
        await initializeWithSampleData();
      } else {
        errorMessage += "Please make sure the backend service is running.";
      }
      
      addMessage(errorMessage, 'bot');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const quickQuestions = [
    "How can I save water at home?",
    "What are the best water conservation techniques?",
    "How do I maintain clean drinking water?",
    "Why is water sanitation important?",
    "What are water-efficient gardening tips?"
  ];

  const StatusIndicator = () => (
    <div className="flex items-center gap-2 text-sm">
      {systemStatus === 'ready' ? (
        <><CheckCircle className="w-4 h-4 text-green-500" /> Ready</>
      ) : systemStatus === 'initialized' ? (
        <><AlertCircle className="w-4 h-4 text-yellow-500" /> Initializing</>
      ) : (
        <><AlertCircle className="w-4 h-4 text-red-500" /> Offline</>
      )}
    </div>
  );

  return (
    <div className="min-h-screen water-pattern bg-gradient-to-br from-water-blue-50 to-ocean-teal-50">
      {/* Header */}
      <header className="water-gradient shadow-lg">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <Droplets className="w-8 h-8 text-white animate-water-drop" />
                <div className="absolute -top-1 -right-1">
                  <Sparkles className="w-4 h-4 text-yellow-300" />
                </div>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">AquaBot</h1>
                <p className="text-water-blue-100 text-sm">Water Conservation Assistant</p>
              </div>
            </div>
            <StatusIndicator />
          </div>
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="max-w-4xl mx-auto px-4 py-6">
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/50 overflow-hidden">
          
          {/* Chat Messages */}
          <div className="h-96 overflow-y-auto p-6 space-y-4">
            {messages.map((message) => (
              <div key={message.id} className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className="flex items-start gap-3 max-w-xs md:max-w-md lg:max-w-lg">
                  {message.sender === 'bot' && (
                    <div className="water-gradient p-2 rounded-full flex-shrink-0">
                      <Droplets className="w-5 h-5 text-white" />
                    </div>
                  )}
                  
                  <div className={`${message.sender === 'user' ? 'chat-bubble-user text-white' : 'chat-bubble-bot text-gray-800'} px-4 py-3 shadow-md`}>
                    <p className="text-sm leading-relaxed">{message.text}</p>
                    
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200">
                        <p className="text-xs text-gray-600 mb-2 flex items-center gap-1">
                          <FileText className="w-3 h-3" />
                          Sources:
                        </p>
                        {message.sources.map((source, idx) => (
                          <div key={idx} className="text-xs bg-gray-100 rounded p-2 mb-1">
                            <p className="text-gray-700">{source.preview}</p>
                          </div>
                        ))}
                      </div>
                    )}
                    
                    <p className="text-xs opacity-70 mt-2">
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="flex items-start gap-3">
                  <div className="water-gradient p-2 rounded-full">
                    <Loader2 className="w-5 h-5 text-white animate-spin" />
                  </div>
                  <div className="chat-bubble-bot px-4 py-3 shadow-md">
                    <p className="text-sm text-gray-600">Thinking about your question...</p>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Questions */}
          {messages.length <= 1 && (
            <div className="px-6 pb-4">
              <p className="text-sm text-gray-600 mb-3 flex items-center gap-2">
                <MessageCircle className="w-4 h-4" />
                Try asking about:
              </p>
              <div className="flex flex-wrap gap-2">
                {quickQuestions.map((question, idx) => (
                  <button
                    key={idx}
                    onClick={() => setInputText(question)}
                    className="text-xs bg-water-blue-100 hover:bg-water-blue-200 text-water-blue-800 px-3 py-2 rounded-full transition-colors duration-200"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input Area */}
          <div className="border-t border-gray-200 p-6 bg-white/50">
            <div className="flex items-end gap-3">
              <div className="flex-1">
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask me about water conservation, sanitation, or saving techniques..."
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-water-blue-500 focus:border-transparent resize-none"
                  rows="2"
                  disabled={isLoading}
                />
              </div>
              
              <button
                onClick={handleSendMessage}
                disabled={!inputText.trim() || isLoading}
                className="water-button p-3 rounded-xl text-white disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
            
            <div className="flex items-center justify-between mt-4 text-xs text-gray-500">
              <p>Press Enter to send, Shift+Enter for new line</p>
              {!isInitialized && (
                <button
                  onClick={initializeWithSampleData}
                  disabled={isLoading}
                  className="flex items-center gap-1 text-water-blue-600 hover:text-water-blue-800 disabled:opacity-50"
                >
                  <Upload className="w-3 h-3" />
                  Initialize Knowledge Base
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Info Cards */}
        <div className="mt-8 grid md:grid-cols-3 gap-6">
          <div className="bg-white/60 backdrop-blur-sm rounded-xl p-6 border border-white/50">
            <div className="flex items-center gap-3 mb-3">
              <Droplets className="w-6 h-6 text-water-blue-600" />
              <h3 className="font-semibold text-gray-800">Water Saving</h3>
            </div>
            <p className="text-sm text-gray-600">Learn practical techniques to reduce water consumption at home and in your community.</p>
          </div>
          
          <div className="bg-white/60 backdrop-blur-sm rounded-xl p-6 border border-white/50">
            <div className="flex items-center gap-3 mb-3">
              <Waves className="w-6 h-6 text-ocean-teal-600 animate-wave" />
              <h3 className="font-semibold text-gray-800">Clean Water</h3>
            </div>
            <p className="text-sm text-gray-600">Discover methods to maintain and ensure access to clean, safe drinking water.</p>
          </div>
          
          <div className="bg-white/60 backdrop-blur-sm rounded-xl p-6 border border-white/50">
            <div className="flex items-center gap-3 mb-3">
              <Sparkles className="w-6 h-6 text-yellow-500" />
              <h3 className="font-semibold text-gray-800">Sanitation</h3>
            </div>
            <p className="text-sm text-gray-600">Understand the importance of water sanitation for health and environmental protection.</p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="text-center py-6 text-gray-500 text-sm">
        <p>AquaBot - Educating communities about water conservation • Built with ❤️ for a sustainable future</p>
      </footer>
    </div>
  );
}

export default App;
