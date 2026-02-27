import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { SignedIn, SignedOut, SignInButton, SignUpButton, UserButton } from '@clerk/clerk-react';
import './App.css';
import Landing from './pages/Landing';
import Reports from './pages/Reports';
import Topics from './pages/Topics';
import TopicDetail from './pages/TopicDetail';

function App() {
  return (
    <Router>
      <div className="app-shell">
        {/* Auth Header */}
        <div className="auth-header">
          <SignedOut>
            <div className="auth-buttons">
              <SignInButton mode="modal">
                <button className="auth-btn auth-btn-signin">Sign In</button>
              </SignInButton>
              <SignUpButton mode="modal">
                <button className="auth-btn auth-btn-signup">Sign Up</button>
              </SignUpButton>
            </div>
          </SignedOut>
          <SignedIn>
            <div className="auth-user">
              <UserButton
                appearance={{
                  elements: {
                    avatarBox: { width: '36px', height: '36px' }
                  }
                }}
              />
            </div>
          </SignedIn>
        </div>

        {/* Main Content */}
        <SignedOut>
          <div className="auth-gate">
            <div className="auth-gate-content">
              <div className="auth-gate-icon">🎬</div>
              <h1>Video Summarizer</h1>
              <p className="muted">
                Upload a video file, or paste a Google Drive or YouTube video link,
                kick off processing, and see transcript + key frames + insights.
              </p>
              <p className="auth-gate-cta">Sign in to get started</p>
              <div className="auth-buttons-large">
                <SignInButton mode="modal">
                  <button className="auth-btn-large auth-btn-signin-large">Sign In</button>
                </SignInButton>
                <SignUpButton mode="modal">
                  <button className="auth-btn-large auth-btn-signup-large">Create Account</button>
                </SignUpButton>
              </div>
            </div>
          </div>
        </SignedOut>

        <SignedIn>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/topics" element={<Topics />} />
            <Route path="/topics/:topicId" element={<TopicDetail />} />
          </Routes>
        </SignedIn>
      </div>
    </Router>
  );
}

export default App;
