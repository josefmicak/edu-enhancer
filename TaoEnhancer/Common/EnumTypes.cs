namespace Common
{
    /// <summary>
    /// A class containing enumerable data types that are used throughout the application
    /// </summary>
    public class EnumTypes
    {
        /// <summary>
        /// Platform on which the application is running (Windows/Linux)
        /// </summary>
        public enum Platform
        {
            Windows,
            Linux
        }

        /// <summary>
        /// Role of the user (Student/Teacher/Admin/MainAdmin)
        /// </summary>
        public enum Role
        {
            Student = 1,
            Teacher = 2,
            Admin = 3,
            MainAdmin = 4
        }

        /// <summary>
        /// State of the application (Waiting/Accepted/Rejected)
        /// </summary>
        public enum RegistrationState
        {
            Waiting,
            Accepted,
            Rejected
        }

        /// <summary>
        /// Negative points in the test (Disabled/EnabledForQuestion/Enabled)
        /// </summary>
        public enum NegativePoints
        {
            Disabled = 1,
            EnabledForQuestion = 2,
            Enabled = 3
        }

        /// <summary>
        /// Type of the subquestion (up to 10 possible types of subquestions are implemented in the application)
        /// </summary>
        public enum SubquestionType
        {
            Error = 0, 
            OrderingElements = 1,
            MultiChoiceMultipleCorrectAnswers = 2,
            MatchingElements = 3,
            MultipleQuestions = 4,
            FreeAnswer = 5,
            MultiChoiceSingleCorrectAnswer = 6,
            MultiChoiceTextFill = 7,
            FreeAnswerWithDeterminedCorrectAnswer = 8,
            GapMatch = 9,
            Slider = 10
        }

        /// <summary>
        /// Whether student's answer is correct (Correct/PartiallyCorrect/Incorrect/NotDetermined)
        /// </summary>
        public enum AnswerCorrect
        {
            NotDetermined = 0,
            Correct = 1, 
            PartiallyCorrect = 2,
            Incorrect = 3,
            NotAnswered = 4,
            CannotBeDetermined = 5
        }
    }
}
