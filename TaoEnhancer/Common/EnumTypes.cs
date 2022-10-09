namespace Common
{
    /// <summary>
    /// A class containing enumerable data types that are used throughout the application
    /// </summary>
    public class EnumTypes
    {
        /// <summary>
        /// Platform on which the application is running (Windows/Ubuntu)
        /// </summary>
        public enum Platform
        {
            Windows,
            Ubuntu
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
    }
}
