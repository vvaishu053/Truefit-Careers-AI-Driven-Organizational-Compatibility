import React, { useEffect } from "react";
import { BrowserRouter as Router, Route, Routes, useLocation } from "react-router-dom";
import Navbar from "./components/Navbar";
import Login from "./components/Login";
import Register from "./components/Register";
import Home from "./components/Home";
import ResumeAnalyzer from "./components/ResumeAnalyzer";
import ResumeDisplay from "./components/ResumeDisplay";
import Profile from "./components/Profile";
import BehaviourTest from "./components/BehaviourTest"; 
import PersonalityResult from "./components/PersonalityResult"; 
import JobSearch from "./components/JobSearch";             
import JobDetails from "./components/JobDetails";
import Chatbot from "./components/Chatbot";
import InterviewDashboard from "./components/InterviewDashboard";
import Interview from "./components/Interview";
import InterviewResults from "./components/InterviewResults";
import InterviewReport from "./components/InterviewReport";
import Recommendations from "./components/Recommendations";
import axios from "axios";

axios.defaults.withCredentials = true;

// Scroll helper for hash links
function ScrollToHashElement() {
  const location = useLocation();
  useEffect(() => {
    if (location.hash) {
      const element = document.querySelector(location.hash);
      if (element) element.scrollIntoView({ behavior: "smooth" });
    }
  }, [location]);
  return null;
}

// Layout wrapper to conditionally show Navbar
function Layout({ children }) {
  const location = useLocation();
  const hideNavbar = location.pathname === "/" || location.pathname === "/register";

  return (
    <>
      {!hideNavbar && <Navbar handleLogout={() => window.location.href = "/"} />}
      {children}
    </>
  );
}

function App() {
  return (
    <Router>
      <ScrollToHashElement />
      <Layout>
        <Routes>
          {/* Public pages */}
          <Route path="/" element={<Login />} />
          <Route path="/register" element={<Register />} />

          {/* Pages with Navbar */}
          <Route path="/home" element={<Home />} />
          <Route path="/resume-analyzer" element={<ResumeAnalyzer />} />
          <Route path="/resume-display" element={<ResumeDisplay />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/behaviour" element={<BehaviourTest />} />
          <Route path="/result" element={<PersonalityResult />} />
          <Route path="/search" element={<JobSearch />} />
          <Route path="/job/:job_role/:company_name" element={<JobDetails />} />
          <Route path="/chatbot" element={<Chatbot />} />
          <Route path="/interview-dashboard" element={<InterviewDashboard />} />
          <Route path="/interview" element={<Interview />} />
          <Route path="/interview-results" element={<InterviewResults />} />
          <Route path="/interview-report" element={<InterviewReport />} />
          <Route path="/recommendations" element={<Recommendations />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
