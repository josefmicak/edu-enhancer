﻿using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class Student
    {
        [Key]
        public string Login { get; set; } = default!;
        public string? Email { get; set; }
        public string StudentIdentifier { get; set; } = default!;
        public string FirstName { get; set; } = default!;
        public string LastName { get; set; } = default!;
        public string FullName()
        {
            return FirstName + " " + LastName;
        }
    }
}